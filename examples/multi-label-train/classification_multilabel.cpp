#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

std::vector<std::vector<std::string> >readlabels(std::string label_file_path)
{
    std::ifstream flabel(label_file_path.c_str());
    std::vector<std::vector<std::string> >labels;
    vector<string> label_file;
    string line;
    int pos = label_file_path.find_last_of("/");
    string labeldirectory = label_file_path.substr(0, pos + 1);
    while (!flabel.eof())
    {
        getline(flabel, line);
        label_file.push_back(labeldirectory + line);
    }
    //std::cout << "the labels' channel:" << label_file.size() << std::endl;
    for (int i = 0; i < label_file.size(); i++)
    {
        std::ifstream labelfile(label_file[i].c_str());
        CHECK(labelfile) << "Unable to open labels file " << label_file[i];
        std::vector<std::string> label_array;
        while (std::getline(labelfile, line))
        {
            label_array.push_back(line);
        }
        labels.push_back(label_array);
    }
    return labels;
}

/* Pair (label, confidence) representing a prediction. */
// change: 2 classify  (label1, confidence1) (label2, confidence2)
typedef std::pair<string, float> Prediction;
class Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file,
        const std::string label_file_path);

	std::vector<vector<Prediction> > Classify(const cv::Mat& img, int N = 5);

private:
	std::vector<vector<float> > Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	std::vector<vector<string> > labels_; //multi
};

Classifier::Classifier(const string& model_file,
	const string& trained_file,
    const std::string label_file_path) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	//CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    labels_ = readlabels(label_file_path);
}

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

/* Return the top N predictions. */
std::vector<vector<Prediction> > Classifier::Classify(const cv::Mat& img, int N) {
	std::vector<vector<float> > output = Predict(img);
	vector<vector<Prediction> > predictions;
	for (int i = 0; i < labels_.size(); i++)
	{
		int Ni = std::min<int>(labels_[i].size(), N);
		std::vector<int> maxNi = Argmax(output[i], Ni);
		std::vector<Prediction> predictionsi;
		for (int j = 0; j < Ni; ++j) {
			int idx = maxNi[j];
			predictionsi.push_back(std::make_pair(labels_[i][idx], output[i][idx]));
		}
		predictions.push_back(predictionsi);
	}
	return predictions;
}

std::vector<vector<float> > Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();
	vector<vector<float> > prob_matrix;
	/* Copy the output layer to a std::vector */
	for (int i = 0; i < labels_.size(); i++)
	{
		Blob<float>* output_layeri = net_->output_blobs()[i];
		const float* begini = output_layeri->cpu_data();
		const float* endi = begini + output_layeri->channels();
		std::vector<float> probi(begini, endi);
		prob_matrix.push_back(probi);
	}
	return prob_matrix;
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

    sample_float.convertTo(sample_float, CV_32FC3, 0.0078125, -127.5*0.0078125);
	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_float, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
	if (argc != 5) {
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< " mean.binaryproto labels.txt img.jpg" << std::endl;
		return 1;
	}

	::google::InitGoogleLogging(argv[0]);

	string model_file = argv[1];
	string trained_file = argv[2];
	string label_file_path = argv[3];
	string file = argv[4];
	Classifier classifier(model_file, trained_file,label_file_path);
	std::cout << "---------- Prediction for " << file << " ----------" << std::endl;
	cv::Mat img = cv::imread(file, -1);
	CHECK(!img.empty()) << "Unable to decode image " << file;
	std::vector<vector<Prediction> > predictions = classifier.Classify(img);
	std::cout << "have ruturned classifier.Classify" << std::endl;
	string pretxt;
	for (int t = 0; t < predictions.size(); t++)
	{
		std::cout << "----------labels:" << t << " ------------" << std::endl;
		for (size_t i = 0; i < predictions[t].size(); ++i) {
			Prediction p = predictions[t][i];
			std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
				<< p.first << "\"" << std::endl;
		}
		pretxt = pretxt + " " + predictions[t][0].first;
	}
	cv::putText(img, pretxt, cv::Point(100, 100), 3, 1, CV_RGB(255, 0, 0));
	cv::imshow("img", img);
	cv::waitKey();
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
