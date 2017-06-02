import numpy as np

def naoLinear(x, deriv = False):
    if(deriv== True):
        return (x*(1-x))
    #sinoide sei la como chama essa porra
    return 1/(1+np.exp(-x))


x = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0]])
y = np.array([[0],
              [1],
              [2],
              [3],
              [4],
              [5],
              [6]])


#semente
np.random.seed(1)


#sinapses

syn0 = 2*np.random.random((3,7))-1
syn1 = 2*np.random.random((7,1))-1


#treinamento

for j in xrange(600000):

    #camadas
    l0= x
    l1 = naoLinear(np.dot(l0,syn0))
    l2 = naoLinear(np.dot(l1, syn1))


    #backpropagation
    l2_erro = y - l2

    #para monitorar o erro
    if(j % 10000)==0 :
        print 'Erro: ' + str(np.mean(np.abs(l2_erro)))

    #calculando os deltas
    l2_delta = l2_erro*naoLinear(l2, deriv=True)

    l1_erro = l2_delta.dot(syn1.T)

    l1_delta = l1_erro*naoLinear(l1, deriv=True)

    #atualizando as synapses
    syn1 += l1.T.dot(l2_delta)

    syn0 += l0.T.dot(l1_delta)


print "Resultado depois do treino"

print l2



















