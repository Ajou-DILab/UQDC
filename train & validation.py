def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def evaluate_loss(net, dataloader, epoch, criterion):
    net.eval()

    mean_loss = 0
    count = 0
    val_correct = 0

    with torch.no_grad():
        for it, input in enumerate(tqdm(dataloader)):

            #q_ids, q_mask, a_ids, a_mask, labels = q_ids.to(device), q_mask.to(device), a_ids.to(device), a_mask.to(device), labels.to(device)
            question, answer, labels = input['question'], input['answer'], input['label'].to(device)
            one_hot = one_hot_embedding(labels, 2)
            one_hot = one_hot.to(device)

                # Obtaining the logits from the model
            logits = net(question, answer)
            #mean_loss += criterion(logits.squeeze(-1), labels.float())
            mean_loss += criterion(logits, one_hot.float(), epoch, 2, 30, device)
            _, predicted = torch.max(logits, 1)
            val_correct += (labels == predicted).sum().cpu()
            count += 1

    return mean_loss / count, val_correct

def train_bert(net, criterion, optim, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate):
    best_acc = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 25  # print the training loss 5 times per epoch

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        train_correct = 0
        for it, input in enumerate(tqdm(train_loader)):

            #q_ids, q_mask, a_ids, a_mask, labels = q_ids.to(device), q_mask.to(device), a_ids.to(device), a_mask.to(device), labels.to(device)

            question, answer, labels = input['question'], input['answer'], input['label'].to(device)

            one_hot = one_hot_embedding(labels, 2)
            one_hot = one_hot.to(device)

                # Obtaining the logits from the model
            logits = net(question, answer)
            #loss = criterion(logits.squeeze(-1), labels.float())
            # Computing loss
            loss = criterion(
                logits.squeeze(-1), one_hot.float(), ep, 2, 30, device
            )

            _, predicted = torch.max(logits, 1)
            train_correct += (labels == predicted).sum().cpu()
            #loss = criterion(logits.squeeze(-1), one_hot.float(), ep, 3, 10, device)

            loss.backward()

            optim.step()
            # Updates the scale for next iteration.
            # Adjust the learning rate based on the number of iterations.
            lr_scheduler.step()
            # Clear gradients
            optim.zero_grad()

            running_loss += loss.item()

            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {}"
                      .format(it + 1, nb_iterations, ep + 1, running_loss / print_every))

                running_loss = 0.0

        val_loss, val_acc = evaluate_loss(net, val_loader, ep, criterion)  # Compute validation loss
        print()
        val_acc = val_acc / len(df_val)
        print("Epoch {} complete! Train ACC : {} Validation Loss : {} ACC : {}".format(ep + 1,
                                                                                       train_correct / len(df_train),
                                                                                       val_loss,
                                                                                       val_acc))

        if val_acc > best_acc:
            print("Best validation acc improved from {} to {}".format(best_acc, val_acc))
            print()
            path_to_model = 'models/_lr_{}_val_loss_{}_ep_{}_acc_{:.4f}.pt'.format(lr,
                                                                                   best_acc,
                                                                                   best_ep,
                                                                                   val_acc)
            torch.save(net.state_dict(), path_to_model)
            print("The model has been saved in {}".format(path_to_model))
            best_acc = val_acc
            best_ep = ep + 1
        # Saving the model

    del loss
    torch.cuda.empty_cache()
  
maxlen = 128
bs = 64   # batch size
iters_to_accumulate = 1
lr = 3e-5  # learning rate
epochs = 35  # number of training epochs

model = BERT()
print(model)
train_set = CustomDataset(df_train, maxlen = maxlen)
val_set = CustomDataset(df_val, maxlen = maxlen)

train_loader = DataLoader(train_set,batch_size=bs, num_workers=0, shuffle= True, drop_last= True)
val_loader = DataLoader(val_set,batch_size=bs, num_workers=0, shuffle= False, drop_last= True)

opti = AdamW(model.parameters(), lr=lr, eps = 1e-8)
#opti = torch.optim.Adam(model.parameters(), lr=lr)
#opti = torch.optim.SGD(model.parameters(), lr=lr, momentum= 0.9, nesterov= True, weight_decay = 1e-6)
num_warmup_steps = 0 # The number of steps for the warmup phase.
num_training_steps = epochs * len(train_loader)  # The total number of training steps
t_total = (len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation

lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps,
                                               num_training_steps=t_total)
criterion = edl_mse_loss
model.to(device)
train_bert(model, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate)
