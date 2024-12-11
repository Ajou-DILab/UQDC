class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model="bert-base-uncased", freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)
        self.fix_annealing_rate = False
        self.kl_scaling_factor = 1.0


        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.act = nn.GELU()
        self.dropout = nn.Dropout(p = 0.1)
        self.cls_layer = nn.Linear(768, 2)

    def forward(self, q_ids, q_mask, token, p_label):
        output = self.bert_layer(q_ids.to(device), q_mask.to(device), token.to(device))
        pooled_output = output["pooler_output"]
        #dropout_rate = random.choice(self.rate)
        logits = self.cls_layer(self.dropout(pooled_output))

        """logit_list = []
        for dropout in self.dropout_layer:
            logit = self.cls_layer(dropout(pooled_output))
            logit_list.append(logit)

        logits = sum(logit_list) / len(self.rate)"""
        alpha = F.softplus(logits) + 1
        #logits = self.cls_layer(self.dropout(pooled_output))

        # only used logits
        return logits, alpha
