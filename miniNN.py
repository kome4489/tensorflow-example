import random
import math
    

def main():
# 訓練データ
    inputs= []

    results= []

    inputs.append([
                   1,
                 1,  1,
                   0,
                 1,  1,  
                   1,
                ])

    results.append([1, #0
                    0, # 1
                    0, # 2
                    0, # 3
                    0, # 4
                    0, # 5
                    0, # 6
                    0, # 7
                    0, # 8
                    0, # 9
                  ])
                  


    inputs.append([
                   0,
                 0,  1,
                   0,
                 0,  1,  
                   0,
                ])

    results.append([0, #0
                    1, # 1
                    0, # 2
                    0, # 3
                    0, # 4
                    0, # 5
                    0, # 6
                    0, # 7
                    0, # 8
                    0, # 9
                  ])
 
    
    inputs.append([
                   1,
                 0,  1,
                   1,
                 1,  0,  
                   1,
                ])

    results.append([0, #0
                    0, # 1
                    1, # 2
                    0, # 3
                    0, # 4
                    0, # 5
                    0, # 6
                    0, # 7
                    0, # 8
                    0, # 9
                  ])

    inputs.append([
                   1,
                 0,  1,
                   1,
                 0,  1,  
                   1,
                ])

    results.append([0, #0
                    0, # 1
                    0, # 2
                    1, # 3
                    0, # 4
                    0, # 5
                    0, # 6
                    0, # 7
                    0, # 8
                    0, # 9
                  ])
    inputs.append([
                   0,
                 1,  1,
                   1,
                 0,  1,  
                   0,
                ])

    results.append([0, #0
                    0, # 1
                    0, # 2
                    0, # 3
                    1, # 4
                    0, # 5
                    0, # 6
                    0, # 7
                    0, # 8
                    0, # 9
                  ])

    inputs.append([
                   1,
                 1,  0,
                   1,
                 0,  1,  
                   1,
                ])

    results.append([0, #0
                    0, # 1
                    0, # 2
                    0, # 3
                    0, # 4
                    1, # 5
                    0, # 6
                    0, # 7
                    0, # 8
                    0, # 9
                  ])

    inputs.append([
                   1,
                 1,  0,
                   1,
                 1,  1,  
                   1,
                ])
    results.append([0, #0
                    0, # 1
                    0, # 2
                    0, # 3
                    0, # 4
                    0, # 5
                    1, # 6
                    0, # 7
                    0, # 8
                    0, # 9
                  ])

    inputs.append([
                   1,
                 0,  1,
                   0,
                 0,  1,  
                   0,
                ])
    results.append([0, #0
                    0, # 1
                    0, # 2
                    0, # 3
                    0, # 4
                    0, # 5
                    0, # 6
                    1, # 7
                    0, # 8
                    0, # 9
                      ])

    inputs.append([
                   1,
                 1,  1,
                   1,
                 1,  1,  
                   1,
                ])
    results.append([0, #0
                    0, # 1
                    0, # 2
                    0, # 3
                    0, # 4
                    0, # 5
                    0, # 6
                    0, # 7
                    1, # 8
                    0, # 9
                  ])

                  
    inputs.append([
                   1,
                 1,  1,
                   1,
                 0,  1,  
                   1,
                ])

    results.append([0, #0
                    0, # 1
                    0, # 2
                    0, # 3
                    0, # 4
                    0, # 5
                    0, # 6
                    0, # 7
                    0, # 8
                    1, # 9
                  ])
                  

    (input_to_hidden_layer_weights,hidden_layers_weights,hidden_to_output_layer_weights)=training(inputs,results)


def training(inputs,results):

    #input -> hidden layer
    # layer1_weights= [        [ 0.0] * 9,
    #                   [ 0.0] * 9,   [ 0.0] * 9,
    #                          [ 0.0] * 9,
    #                   [ 0.0] * 9,     [ 0.0] * 9,
    #                        [ 0.0] * 9,
    #                ]
    # layer1_bias =[ 0.5 , 0.5 , 0.5 ,
    #                0.5 , 0.5 , 0.5 ,
    #                0.5 , 0.5 , 0.5 ,
    #              ]
                    
    #hidden layer->result
    input_to_hidden_layer_weights= None

    hidden_layers_weights = None
    
    hidden_to_output_layer_weights = None

    NN_score =0

    #トレーニング
    for i in range(10000):
    #新し個体を生成する
        new_input_to_hidden_layer_weights = []
        for x in range(10):  
            inner =[]
            new_input_to_hidden_layer_weights.append(inner)
            for y in range(8): #weights + bias
              inner.append( random.random()*2-1)

        new_hidden_layers_weights = []
        for layer_index in range(1):
            new_hidden_layers_weights.append([])
            for x in range(10):  
                inner =[]
                new_hidden_layers_weights[layer_index].append(inner)
                for y in range(11): #weights + bias
                    inner.append( random.random()*2-1)

        new_hidden_to_output_layer_weights = []
        for x in range(10):
            inner =[]
            new_hidden_to_output_layer_weights.append(inner)
            for y in range(11): #weights + bias
                inner.append( random.random()*2-1)

        new_NN_score = 0
        right_answer_inputs=[]
        right_answers_digit =[]
        right_answers =[]
        #　inputs をNNに渡して、全訓練データに対して結果を算出してくれる
        for index,one_input in enumerate(inputs):

            new_ret_val = run_NN(new_input_to_hidden_layer_weights,new_hidden_layers_weights,new_hidden_to_output_layer_weights,one_input)
            if  right_answer(new_ret_val,results[index]) >0 :
              right_answer_inputs.append(one_input)
              right_answers_digit.append(new_ret_val.index(max(new_ret_val)))
              right_answers.append(new_ret_val)
              new_NN_score+=1

        if(new_NN_score>NN_score):
            print("traning %d"%i )
            print("old score =%d, new score= %d"%(NN_score,new_NN_score)) 
            print("input :") 
            print(right_answer_inputs)
            print("result  :") 
            print(right_answers_digit)
            print("full result  :") 
            print(right_answers)

            input_to_hidden_layer_weights=new_input_to_hidden_layer_weights
            hidden_layers_weights=new_hidden_layers_weights
            hidden_to_output_layer_weights = new_hidden_to_output_layer_weights

            # layer99_bias=new_layer99_bias
            NN_score =new_NN_score

    return (input_to_hidden_layer_weights,hidden_layers_weights,hidden_to_output_layer_weights)

def right_answer(answer,correct):
    #gap =0.5
    if(correct.index(max(correct)) == answer.index(max(answer))):
        return 1

    # if( abs(correct [0] - answer[0]) < gap and  abs(correct [1] - answer[1]) < gap and abs(correct [2] - answer[2]) < gap and abs(correct [3] - answer[3]) < gap ):
    #     return 1

    return 0

    
def run_NN(new_input_to_hidden_layer_weights,new_hidden_layers_weights,new_hidden_to_output_layer_weights,one_input):
  mid_layer=one_input
  
  mid_layer = run_one_layer( new_input_to_hidden_layer_weights ,mid_layer)
  
  for hidden_layer_weights in  new_hidden_layers_weights :
    mid_layer=run_one_layer( hidden_layer_weights ,mid_layer)

  return run_one_layer( new_hidden_to_output_layer_weights ,mid_layer)



def run_one_layer( layer_weights ,input_data):
    ret_val = [0.0] *(len(layer_weights))
    for x in range(len(layer_weights)):
        sum = 0.0
        for y in range(len(input_data)):
            sum+=input_data[y] * layer_weights[x][y]
        sum =sum - layer_weights[x][len(layer_weights[0])-1]
        ret_val[x]  = activation_function(sum)
    return ret_val

def activation_function(x):
    #sigmoid
    return 1 / (1 + math.exp(-x))

if __name__== "__main__":
  main()