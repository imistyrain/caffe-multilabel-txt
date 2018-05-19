import numpy as np
import matplotlib.pyplot as plt
import sys,os,argparse
import time
caffe_root = 'D:/CNN/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

modeldir="../"
model_def =modeldir+ '/deploy.prototxt'
model_weights = modeldir+ '/trainedmodels/AlexNet_iter_1000.caffemodel'
labelfile_path=modeldir+ '/labels/labels.txt'

def loadlabels(labelfile_path):
    labels=[]
    with open(labelfile_path) as f:
        while True:
            line=f.readline()
            if not line:
                break
            lbp=os.path.dirname(labelfile_path)+"/"+line.split()[0]
            label = np.loadtxt(lbp, str, delimiter='\t')
            labels.append(label)
    return labels

def loadmean(meanprotopath):
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(open(meanprotopath, 'rb').read())   
    return np.array(caffe.io.blobproto_to_array(blob))[0]

def gettransformer(net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    #transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    #transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    return transformer

labels=loadlabels(labelfile_path)
net = caffe.Net(model_def,model_weights,caffe.TEST)
#mu = loadmean(meanprotofile_path).mean(1).mean(1) #print 'mean-subtracted values:', zip('BGR', mu)
transformer=gettransformer(net)
net.blobs['data'].reshape(1,3,227,227)

def testimage(args,imgfilepath):
    image=caffe.io.load_image(imgfilepath)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    index=0
    strtitle=""
    for op in output:
        prob=net.blobs[op]
        output_prob = prob.data[0]
        print(op+":",output_prob.argmax(),labels[index][output_prob.argmax()], output_prob[output_prob.argmax()])
        strtitle+=labels[index][output_prob.argmax()]+":"+str(output_prob[output_prob.argmax()])
        if args.print5scores:
            top_inds = output_prob.argsort()[::-1][:5]
            zip(output_prob[top_inds],labels[index][top_inds])
        index+=1
    if args.show_resultimage:
        plt.imshow(image)
        plt.title(strtitle)
        plt.show()

def testdir(args):
    caffe.set_device(0)
    if args.gpu:
        caffe.set_mode_gpu()
    files=os.listdir(args.imgdir)
    for file in files:
        print(file)
        imgfilepath=args.imgdir+"/"+file
        testimage(args,imgfilepath)

def main(args):
    testdir(args)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",default=True,help="Use GPU")
    parser.add_argument("--imgdir",default="../ZnCar/Train",help="Directory of images to classify")
    parser.add_argument("--print5scores",default=False,help="Show top 5 socres")
    parser.add_argument("--show_resultimage",default=True,help="Show result image")
    args = parser.parse_args()
    main(args)