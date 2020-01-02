import pandas as pd
import numpy


#PREPROCESSING THE CSV FILE
def dataframe(path=''):

    df = pd.read_csv(path,usecols=['Price','Open','High','Low','Vol'])
    df = df.apply(lambda x: x.str.replace(',',''))

    #Converting M,K and B into numbers
    df['Vol'] = df['Vol'].replace('-','0')
    df['Vol'] = df['Vol'].replace(r'[KMB]+$','',regex=True).astype(float) * df['Vol'].str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M','B'], [10**3, 10**6, 10**9]).astype(int)

    df = df.astype(float)
    return df


def mean_normalize(df):
    normalized_df=(df-df.mean())/df.std()
    return normalized_df


def min_max_normalize(df):
    normalized_df = (df-df.min())/(df.max()-df.min())
    return normalized_df

def de_normalize_minmax(df):
    de_normalized = (df_max-df_min)*df + df_min
    return de_normalized

#print(sheet)

#NEURAL NETWORK CLASS
class feedforwardclass:
    def  __init__(self ,inputs,n_hidden,output):
        self.inputs=inputs
        self.n_hidden=n_hidden
        self.output=output

        #self.d_out=[]
        self.learning_rate=0.01




#INPUT FUNCTION
def inputf(rows,column):
    dictionary={}
    for i in range(rows-1,0,-1):
        into=[]
        for j in range(column):
            into.append(sheet[i,j])
            #print(sheet[i,j])
            intarr=numpy.matrix(numpy.asarray(into))
        intarr=intarr.reshape(column,1)
        dictionary['%s'%(row-i)]=intarr
    return dictionary
    
    
#derivative of sigmoid
def dsigmoid(y):
    z= numpy.multiply(y,(1-y))
    return z
#derivative of hyperbolic tan
def dtanh(y):
    return(1.0-(numpy.power(y,2)))

#derivative of relu
def  drelu(y):
    x=y
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if x[i,j]==0:
                x[i,j]=0
            else:
                x[i,j]=1
    return x
#derivative of leaky relu
def dlrelu(y):
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i,j]<0:
                y[i,j]=0.01
            #elif y[i,j]==0:
                #y[i,j]=0
            else:
                y[i,j]=1
    return y
    
#hyperbolic tangent
def hypertan(x):
    for i in range (x.shape[0]):
        for j in range (x.shape[1]):
            x[i,j]=(1-numpy.exp(-2*x[i,j]))/(1+numpy.exp(-2*x[i,j]))
    return x

#inverse tangent
def arctan(x):
    for i in range (x.shape[0]):
        for j in range (x.shape[1]):
            x[i,j]=math.atan(x[i,j])
    return x

#rectified linear unit
def relu(x):
    z=x
    z=numpy.maximum(0,z)
    return z


#leaky relu
def Lrelu(matrix):
    for i in range (matrix.shape[0]):
        for j in range (matrix.shape[1]):
            if matrix[i,j]<0:
                matrix[i,j] = numpy.multiply(0.01,matrix[i,j])
            else:
                matrix[i,j] = matrix[i,j]
    return matrix

#gaussian
def gauss(x):
    for i in range (x.shape[0]):
        for j in range (x.shape[1]):
            x[i,j]=numpy.exp(-(numpy.power(x[i,j],2)))
    return x

#mod function
def mod(x):
    if x<0:
        return (-x)
    else:
        return x

#softmax
def softmax(x):
    for i in range (x.shape[0]):
        for j in range (x.shape[1]):
            x[i,j] = x[i,j]/(1+mod(x[i,j]))
    return x

# derivative of soft max
def dsoftmax(y):
    return (numpy.multiply(y,(1-y)))
    
#SIGMOID    
def sigmoid(x):
    z = 1/(1 + numpy.exp(-x))
    return (z)

#MEAN ABSOLUTE ERROR PERCENT
def mean_absolute_error_percent(observed,predicted):
	summation = 0  #variable to store the summation of differences
#	print(observed)
#	print(predicted)
	n = len(observed) #finding total number of items in list
	for i in range (0,n):  #looping through each element of the list
  		difference = (observed[i] - predicted[i])/observed[i]  #finding the difference between observed and predicted value
  		squared_difference = mod(difference)  #taking square of the differene 
  		summation = summation + squared_difference  #taking a sum of all the differences
	MSE = (100*summation)/n
	return MSE

#DROPOUT IMPLEMENTATION
def dropout(prob):
    drop={}
    for i in range(1,ff.n_hidden+1):
        drop['drop%s'%i]=numpy.random.binomial(1,prob,size=hid_dict['hidden%s'%i])
        drop['drop%s'%i]=numpy.matrix(drop['drop%s'%i]).reshape(hid_dict['hidden%s'%i],1)
        #print(drop['drop%s'%i])
    return drop

def  d_outf(output):
    n=output
    d_out=[]
    for i in range(n):
        d_out.append(int(input()))
    d_mat=numpy.matrix(numpy.asarray(d_out))
    return (d_mat)

#function to convert array into matrix of single column
def convertmatrix (input_arr):
    b=[]
    for i in range(len(input_arr)):
        a=[]
        for j in range(1):
            a.append(input_arr[i])
        b.append(a)
    return b  

sheet = dataframe("Data2.csv")
sheet_copy = sheet

sheet_copy = min_max_normalize(sheet_copy)
df_min = sheet.min()[2]
df_max = sheet.max()[3]
#print(df_max,df_min)
sheet = min_max_normalize(sheet)
column = sheet.shape[1]
row = sheet.shape[0]


print(" number of inputs")
inp = column
print(inp)
epoch = row

sheet=numpy.matrix(sheet)  

#CHOICES FOR ACTIVATION FUNCTION
choices=[]
    
print("enter the number of hidden layers")
hid=int(input())
ff=feedforwardclass(inp,hid,1)


#hidden layer input
hid_dict={}
for i in range(1,ff.n_hidden+1):
    print("enter the number of neuron in {} hidden layer".format(i))
    hid_dict['hidden%s'%i]=int(input())


    print("select activation function for this layer:-\n0.hypertan\n1.RELU\n2.Lrelu\n3.softmax\n4.sigmoid\n")
    choice = int(input())
    if choice>=0 and choice<=4:
        choices.append(choice)
    else:
        print("Invalid Choice, RELU by default")
        choices.append(4)

#Activation FUnction for output layer
print("select activation function for output layer:-\n0.hypertan\n1.RELU\n2.Lrelu\n3.softmax\n4.sigmoid\n")
choice = int(input())
if choice>=0 and choice<=4:
    choices.append(choice)
else:
    print("Invalid Choice, RELU by default")
    choices.append(4)

dactivation_function = [dtanh,drelu,dlrelu,dsoftmax,dsigmoid]
activation_function = [hypertan,relu,Lrelu,softmax,sigmoid]
#print(activation_function)

#dropout fuctionality
print("enter probability of drop out")
prob=float(input())





#weights selection
weight_dict={}
weight_dict['weight_ih']=numpy.matrix(numpy.random.random((hid_dict['hidden1'],ff.inputs)))

for i in range(1,ff.n_hidden):
    weight_dict['weight_%s'%(i+1)]=numpy.matrix(numpy.random.random((hid_dict['hidden%s'%(i+1)],hid_dict['hidden%s'%i])))
weight_dict['weight_ho']=numpy.matrix(numpy.random.random((ff.output,hid_dict['hidden%s'%ff.n_hidden])))

#bias selection
bias_dict={}
for i in range(1,ff.n_hidden+1):
    bias_dict['biash%s'%i]=numpy.matrix(numpy.random.random((hid_dict['hidden%s'%i],1)))
bias_dict['biaso']=numpy.matrix(numpy.random.random((ff.output,1)))


#to take input of desired output   


#training function
def  training(inputs,d_output):

    #copy of feedforward function
    inputs=inputs
    hiden_dict={}
    drop=dropout(prob)
    
    hiden_dict['hiden1']= weight_dict['weight_ih'].dot(inputs)
    hiden_dict['hiden1']=hiden_dict['hiden1']+bias_dict['biash1']
    hiden_dict['hiden1']=activation_function[choices[0]](hiden_dict['hiden1'])
    #forward dropout
    hiden_dict['hiden1']=numpy.multiply(hiden_dict['hiden1'],(drop['drop1']))

    for i in range(2,ff.n_hidden+1):
        
        hiden_dict['hiden%s'%i]= weight_dict['weight_%s'%i].dot(hiden_dict['hiden%s'%(i-1)])
        hiden_dict['hiden%s'%i]=hiden_dict['hiden%s'%i]+bias_dict['biash%s'%i]
        hiden_dict['hiden%s'%i]= activation_function[choices[i-1]](hiden_dict['hiden%s'%i])
        #forward dropout
        hiden_dict['hiden%s'%i]=numpy.multiply(hiden_dict['hiden%s'%i],(drop['drop%s'%i]))
        #print(hiden_dict['hiden%s'%i])

    output=(weight_dict['weight_ho']).dot(hiden_dict['hiden%s'%ff.n_hidden])
    output=output+bias_dict['biaso']
    output=activation_function[choices[-1]](output)

    print("training output{}".format(de_normalize_minmax(output)))
    #print(type(output))

    
    #output error
    output_error=numpy.empty([output.shape[0],output.shape[1]])
    output_error=numpy.subtract(d_output,output)
    #transpose of matrix
    weight_tran={}
    hidden_tran={}
    weight_tran['ho']=(weight_dict['weight_ho']).transpose()
    for i in range(1,ff.n_hidden):
        weight_tran['h%s'%i]=(weight_dict['weight_%s'%(i+1)]).transpose()
    for i in range(1,ff.n_hidden+1):
        hidden_tran['%s'%i]=(hiden_dict['hiden%s'%i]).transpose()
    inputs_trans=inputs.transpose()
    #inputs_trans=inputs.transpose()
    
    #hidden error
    hidden_error={}
    hidden_error['%s'%ff.n_hidden]=(weight_tran['ho']).dot(output_error)
    for i in range((ff.n_hidden-1),0,-1):
        hidden_error['%s'%i]=(weight_tran['h%s'%(i)]).dot(hidden_error['%s'%(i+1)])

    weight_delta={}
    gradient={}
    #gradient of output
    gradient['o']=dactivation_function[choices[-1]](output)
    gradient['o']=numpy.multiply(gradient['o'],output_error)
    gradient['o']=gradient['o']*ff.learning_rate
    #delta of output layer
    weight_delta['ho']=gradient['o'].dot(hidden_tran['%s'%ff.n_hidden])
    weight_dict['weight_ho']=weight_dict['weight_ho']+weight_delta['ho']
    
    
    #gradient of hidden
    for i in range(ff.n_hidden,1,-1):
        gradient['h%s'%i]=dactivation_function[choices[i-1]](hiden_dict['hiden%s'%i])
        gradient['h%s'%i]=numpy.multiply(gradient['h%s'%i],hidden_error['%s'%i])
        gradient['h%s'%i]=gradient['h%s'%i]*ff.learning_rate
        #backward dropout
        gradient['h%s'%i]=numpy.multiply(gradient['h%s'%i],drop['drop%s'%(i)])
        #delta of output layer
        weight_delta['h%s'%i]=gradient['h%s'%i].dot(hidden_tran['%s'%(i-1)])
        weight_dict['weight_%s'%i]=weight_dict['weight_%s'%i]+weight_delta['h%s'%i]

    gradient['h1']=dactivation_function[choices[0]](hiden_dict['hiden1'])
    gradient['h1']=numpy.multiply(gradient['h1'],hidden_error['1'])
    gradient['h1']=gradient['h1']*ff.learning_rate
    #backward dropout
    gradient['h1']=numpy.multiply(gradient['h1'],drop['drop1'])
    #delta of output layer
    weight_delta['ih']=gradient['h1'].dot(inputs_trans)
    weight_dict['weight_ih']=weight_dict['weight_ih']+weight_delta['ih']

    return 0
                
#feedforward algo
def feedforward(input_arr):
    inputs=input_arr
    hiden_dict={}
    
    hiden_dict['hiden1']= weight_dict['weight_ih'].dot(inputs)
    hiden_dict['hiden1']=hiden_dict['hiden1']+bias_dict['biash1']
    hiden_dict['hiden1']=activation_function[choices[0]](hiden_dict['hiden1'])

    for i in range(2,ff.n_hidden+1):
        
        hiden_dict['hiden%s'%i]= weight_dict['weight_%s'%i].dot(hiden_dict['hiden%s'%(i-1)])
        hiden_dict['hiden%s'%i]=hiden_dict['hiden%s'%i]+bias_dict['biash%s'%i]
        hiden_dict['hiden%s'%i]=activation_function[choices[i-1]](hiden_dict['hiden%s'%i])


    foutput=(weight_dict['weight_ho']).dot(hiden_dict['hiden%s'%ff.n_hidden])
    foutput=foutput+bias_dict['biaso']
    foutput=activation_function[choices[-1]](foutput)
    return (foutput)

def split_validate(k,training_epoch):
    validate_data = {}
    size = training_epoch//k
    for i in range(k):
        start = size*(i)
        end = start + size 
        validate_data['%s'%i] = sheet_copy.iloc[start:end]
    return validate_data


def main():
    #d_output=[1]
    n = epoch
    print("epoch for training = {}".format(n))
    
    into_dict=inputf(n,column)
    #print(into_dict)
    t_percent = int(input("Enter training percentage="))
    
    test_percent = 100 - t_percent 

    print("Test percentage = {}".format(test_percent))

    training_epoch = (n * t_percent)//100
    K = t_percent//10
    test_epoch = (n * test_percent)//100
    size = training_epoch//K


    observed_values = []
    predicted_values = []
    #training phase
   
    for i in range (training_epoch):
        #for j in range(5):
            d_output=into_dict['%s'%(i+2)]
            d_output=d_output[1,0]
            #d_output=numpy.matrix(d_output[0][0])
            training(into_dict['%s'%(i+1)],d_output)
            print(n-i)
    

    validate_data = split_validate(K,training_epoch)

    for i in range(len(validate_data)):
    	for j in range(len(validate_data)):

    		if i!=j:
    			df = validate_data['%s'%j]
    			#df = numpy.matrix(df)
    			for k in range(df.shape[0]-1):
    				training(numpy.matrix(df.iloc[k]).reshape(column,1),df.iloc[k+1][1])
    				print(k)

    		if i==j:
    			df_t = validate_data['%s'%j]
    			for k in range(df_t.shape[0]-1):
    				result = feedforward(numpy.matrix(df_t.iloc[k]).reshape(column,1))
    				print("test output {}".format(de_normalize_minmax(result)))
    #VALIDATION PHASE

    #for i in range(training_epoch,training_epoch + validation_epoch,1):

    #TESTING PHASE
    for i in range(training_epoch,n-1,1):

        out=feedforward(into_dict['%s'%i])
        observed = into_dict['%s'%(i+1)]
        observed = observed[1,0]
        #print(out)
       	observed_values.append(de_normalize_minmax(observed))
       	predicted_values.append(de_normalize_minmax(out)[0,0])
 
        print("test output {}".format(de_normalize_minmax(out)))
        print(n-i)


    error_percent = mean_absolute_error_percent(observed_values,predicted_values)
    print("error = {}%".format(error_percent))
        #print("Enter Price, Open Price, High and low price and volume")
    # values = []

    # for i in range(5):
    #     values.append(float(input()))

    # values = numpy.matrix(values).reshape(5,1)
    # values = min_max_normalize(values)
    # out = feedforward(values)
    
    
if __name__=='__main__':
    main()


