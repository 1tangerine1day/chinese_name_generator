# A good name is all you need 

presentation : https://drive.google.com/file/d/1-9wwoZZPWz_EVhi28o-vo9GJOOkK69wL/view

## example
* way 1

        input:鍾 target:佳 
        input:佳 target:紋 
        input:紋 target:。 

* way 2

        input:鍾 target:鍾佳
        input:鍾佳 target:鍾佳紋
        input:鍾佳紋 target:鍾佳紋。


## model 

    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNN, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.gru = nn.GRU(self.input_size, self.hidden_size,dropout = 0.5)
            self.h2o = nn.Linear(self.hidden_size, self.output_size)
            self.softmax = nn.LogSoftmax() 
        def forward(self, input):
            out,_ = self.gru(input)
            out = self.h2o(out).squeeze(0).squeeze(0)
            output = self.softmax(out)
            return output
        def initHidden(self):
            return torch.zeros(1, self.hidden_size)
            
## char embedding

    import pickle
    f=  open ('./char_embedding.pkl', 'rb')
    dict_char = pickle.load(f)
    f.close()

## data

crawler data from https://findbiz.nat.gov.tw/fts/query/QueryBar/queryInit.do;jsessionid=F906D81F3071A0B848AC0633D47165D7

