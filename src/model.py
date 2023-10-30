from unicodedata import bidirectional
import torch
from torch import dropout, log_softmax, nn
from torch.nn import functional as F
class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        """_summary_

        Args:
            num_chars (int): Maximum Number of Char
        """
        super(CaptchaModel, self).__init__()
        # CNN
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3,3), padding=(1,1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3,3), padding=(1,1))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.linear_1 = nn.Linear(1152, 64)
        self.drop_1 = nn.Dropout(0.2)
        
        #LSTM
        self.gru = nn.GRU(64,32, bidirectional=True, num_layers=2, dropout=0.25)
        self.output = nn.Linear(64, num_chars + 1) #nu_chars + 1 as need to include unknowns
        
    def forward(self, images, targets = None):
        bs, c, h, w = images.size()
        print(bs, c, h, w)
        
        x = F.relu(self.conv_1(images))
        print(x.size())
        x = self.max_pool_1(x)
        print(x.size())
        x = F.relu(self.conv_2(x))
        print(x.size())
        x = self.max_pool_2(x) #[bs, 64, 18, 75] - bs, c, h (feature), w (time-stamp)
        print(x.size())
        #(batch_size, c, h, w) -> (batch_size, w, c, h)
        x = x.permute(0, 3, 1, 2) #[bs, w=75, 64, 18]
        print(x.size())
        x = x.view(bs, x.size(1), -1) #(batch_size, w, c*h) = [bs, 75, 1152]
        print(x.size())
        x = self.linear_1(x) #reduce from number of filter from 1152 to 64
        x = self.drop_1(x) #[bs, 75, 64] #we have 75 timestamp, and each timestamp has 64 values
        print(x.size())
        
        # LSTM
        x, _ = self.gru(x)
        print(x.size())
        x = self.output(x)
        print(x.size()) #[bs, 75, 20]: for each timestamp, we will have 20 output
        
        x = x.permute(1, 0, 2) #[75, bs, 20]
        #print(x.size())
        if targets is not None:
            #CTC takes log_softmax
            log_softmax_values = F.log_softmax(x,2) #x = output, dim=2 as 20 probs at index 2 in [bs, 75, 20]
            input_lengths = torch.full(
                size=(bs, ),
                fill_value = log_softmax_values.size(0),
                dtype=torch.int32
            )
            print(input_lengths) #tensor([75], dtype=torch.int32)
            target_lengths = torch.full(
                size=(bs, ),
                fill_value = targets.size(1), #5: output_size = 5
                dtype=torch.int32
            )
            print(target_lengths) #tensor([5], dtype=torch.int32)
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values,
                targets,
                input_lengths,
                target_lengths
            )
            return x, loss
        return x, None #prediction & loss
    
if __name__ == "__main__":
    cm = CaptchaModel(19) #19 different chars
    img = torch.rand(5,3, 75, 300)
    target = torch.randint(1,20, (5,5))
    x, loss = cm.forward(img, target)
        
        