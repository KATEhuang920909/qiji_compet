# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn

from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss
from paddlenlp.utils.tools import compare_version

if compare_version(paddle.version.full_version, "2.2.0") >= 0:
    # paddle.text.ViterbiDecoder is supported by paddle after version 2.2.0
    from paddle.text import ViterbiDecoder
else:
    from paddlenlp.layers.crf import ViterbiDecoder
class ErnieGRUCRF(nn.Layer):
    def __init__(self,Ernie,gru_hidden_size=300,
                num_class=2,
                crf_lr=100):
        super().__init__()
        self.num_classes = num_class
        self.Ernie = Ernie
        self.gru = nn.GRU(self.Ernie.config["hidden_size"],
                          gru_hidden_size,
                          num_layers = 2,
                          direction='bidirect')
        self.fc = nn.Linear(gru_hidden_size*2,num_class+2)
        self.crf = LinearChainCrf(self.num_classes)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self,input_ids,token_type_ids,lengths=None,labels=None):
        encoder_output,_ = self.Ernie(input_ids, token_type_ids = token_type_ids)
        gru_output, _ = self.gru(encoder_output)
        emission = self.fc(gru_output)
        if labels is not None:
            loss = self.crf_loss(emission, lengths, labels)
            return loss
        else:
            _,prediction = self.viterbi_decoder(emission, lengths)
            return prediction
