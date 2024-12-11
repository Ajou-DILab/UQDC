def evaluate_loss(net, dataloader, epoch, criterion):
    net.eval()

    mean_loss = 0
    count = 0
    val_correct = 0

    with torch.no_grad():
        for it, (p_ids, p_attn, p_token, p_label) in enumerate(tqdm(dataloader)):
            # q_ids, q_mask, a_ids, a_mask, labels = q_ids.to(device), q_mask.to(device), a_ids.to(device), a_mask.to(device), labels.to(device)

            # Obtaining the logits from the model
            p_one_hot = one_hot_embedding(p_label, 2)
            p_one_hot = p_one_hot.to(device)
            p_label = p_label.to(device)
            logits, alpha = net(p_ids, p_attn, p_token, p_one_hot)

            mean_loss += criterion(logits.squeeze(-1), p_one_hot.float())
            #mean_loss = torch.mean(loss)

            #p_predicted = (alpha.view(-1) >= 0.5).int()
            _, p_predicted = torch.max(alpha, 1)



            val_correct += (p_label == p_predicted).sum().cpu()
            count += 1

    return mean_loss / count, val_correct
