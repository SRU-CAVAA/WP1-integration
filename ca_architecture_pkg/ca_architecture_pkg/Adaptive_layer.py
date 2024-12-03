import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class AdaptiveLayer(object):

    def __init__(self):
        print('---------- GRAPHIC CARD INFO ----------')
        print('Graphic card: ' + str(torch.cuda.get_device_name(0)))
        print('Available: ' + str(torch.cuda.is_available()))

        self.trial_frame_list = []
        self.trial_AE_activation_list = []
        self.trial_homeostate_list = []
        self.history = []
        self.ext_history = []

    def load_AE_model(self, model, n_hidden, motivational_AE, motiv_I_len=0):
        if motivational_AE == True:
            self.AE_model = Motivational_Conv_AE(n_hidden=n_hidden, motiv_I_len=motiv_I_len).to('cuda')
        else:
            self.AE_model = Conv_AE(n_hidden=n_hidden).to('cuda')
        self.AE_model.load_state_dict(torch.load(model))

    def motivation_augmentation(self, homeo_state, motiv_I_len_per_need):
        self.augmented_homeostasis = []
        self.augmented_homeostasis_1D = []

        for i in range(len(homeo_state)):
            augmented_ext_input = []
            if homeo_state[i] == min(homeo_state): motiv_att_value = 1
            else: motiv_att_value = 0
            self.augmented_homeostasis.append([motiv_att_value] * motiv_I_len_per_need)
        
        for i in range(len(self.augmented_homeostasis)):
            self.augmented_homeostasis_1D.extend(self.augmented_homeostasis[i])
        self.augmented_homeostasis_1D = np.array([self.augmented_homeostasis_1D])


    def encode(self, frame, motivational_AE, homeo_state, motiv_I_len_per_need, STM_limit):
        frame = np.transpose(frame, (2,0,1))
        if np.max(frame) > 1:
            normalized_frame = frame/255.

        frame_tensor = torch.Tensor(normalized_frame).to('cuda')
        
        self.trial_frame_list.append(frame)
        if len(self.trial_frame_list) > STM_limit:
            self.trial_frame_list.pop(0)

        if motivational_AE == True:
            self.motivation_augmentation(homeo_state, motiv_I_len_per_need)
            self.trial_homeostate_list.append(self.augmented_homeostasis_1D[0].tolist())
            motiv_state_tensor = torch.tensor(self.augmented_homeostasis_1D, dtype=torch.float32).to('cuda')
            embedding = self.AE_model.encoder(frame_tensor, motiv_state_tensor)[0].detach().cpu().numpy()
            #motiv_state_prediction = self.AE_model.encoder(frame_tensor, motiv_state_tensor)[1].detach().cpu().numpy()
            view_prediction, motiv_state_prediction = predict(normalized_frame, self.AE_model, motiv_state=self.augmented_homeostasis_1D)
        else:
            embedding = self.AE_model.encoder(frame_tensor).detach().cpu().numpy()
            view_prediction, _ = predict(normalized_frame, self.AE_model)
        activation = sum(embedding[0])

        self.trial_AE_activation_list.append(activation)
        if len(self.trial_AE_activation_list) > STM_limit:
            self.trial_AE_activation_list.pop(0)
        #print("Mean activation = ", sum(self.trial_AE_activation_list)/len(self.trial_AE_activation_list))

        return embedding[0], activation, view_prediction


    def retrain(self, motivational_AE, learning_rate, alpha, C_factor, batch_size, num_epochs=0):
        dataset = np.array(self.trial_frame_list)
        if np.max(dataset) > 1:
            normalized_dataset = dataset/255.
        print()
        print("----- RETRAINING " + str(len(self.trial_frame_list)) + " FRAMES DURING " + str(num_epochs) + " EPOCHS-----")

        if motivational_AE == True:
            Homeostasis_tensor = torch.tensor(self.trial_homeostate_list, dtype=torch.float32).to('cuda')
        train_loader = create_dataloader(normalized_dataset, batch_size=batch_size)
        if motivational_AE == True:
            history, _, ext_history= train_motivational_autoencoder(self.AE_model, train_loader, external_inputs=Homeostasis_tensor, num_epochs=num_epochs, learning_rate=learning_rate, alpha=alpha, C_factor=C_factor)
        else:
            history, _ =train_autoencoder(self.AE_model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate, alpha=alpha, C_factor=C_factor)

        self.history.extend(history)
        if motivational_AE == True: self.ext_history.extend(ext_history)
        self.trial_frame_list, self. trial_AE_activation_list, self.trial_homeostate_list = [], [], []

        print("RETRAINED WITH FINAL LOSS = ", self.history[-1])
        print()



    def save_model(self, model_path):
        torch.save(self.AE_model.state_dict(), model_path)




        



class Conv_AE(nn.Module):
    def __init__(self, n_hidden):

        super().__init__()

        self.n_hidden = n_hidden
        self.dim1, self.dim2 = 15, 20

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * self.dim1 * self.dim2, n_hidden)

        # Decoder
        self.fc2 = nn.Linear(n_hidden, 64 * self.dim1 * self.dim2)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encoder(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  
        x = x.view(-1, 64 * self.dim1 * self.dim2)  
        x = F.relu(self.fc1(x))
        return x

    def decoder(self, x):
        # Decoder
        x = F.relu(self.fc2(x)) 
        x = x.view(-1, 64, self.dim1, self.dim2) 
        x = F.relu(self.conv4(x)) 
        x = F.relu(self.conv5(x))  
        x = torch.sigmoid(self.conv6(x))  
        return x

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out, h

    def backward(self, optimizer, criterion, x, y_true,C_factor, alpha=0):
        optimizer.zero_grad()

        y_pred, hidden = self.forward(x)

        recon_loss = criterion(y_pred, y_true)

        # Whitening loss (batch whitening).
        hidden_constraint_loss = 0
        batch_size, hidden_dim = hidden.shape

        # SSCP matrix
        M = torch.mm(hidden.t(), hidden)
        I = torch.eye(hidden_dim, device='cuda')
        C = C_factor*I - M    # C = I - M    
        hidden_constraint_loss = alpha * torch.norm(C) / (batch_size*hidden_dim)
        
        loss = recon_loss + hidden_constraint_loss
        loss.backward()

        optimizer.step()

        return recon_loss.item()


class Motivational_Conv_AE(nn.Module):
    def __init__(self, motiv_I_len, n_hidden=500):
        super(Motivational_Conv_AE, self).__init__()
        self.n_hidden = n_hidden
        self.dim1, self.dim2 = 15, 20

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * self.dim1 * self.dim2 + motiv_I_len, n_hidden)
        self.pred_ext_input = nn.Linear(n_hidden, motiv_I_len)


        # Decoder
        self.fc2 = nn.Linear(n_hidden, 64 * self.dim1 * self.dim2)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encoder(self, x, ext_input):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * self.dim1 * self.dim2)
        # Concatenate external input to the flattened output before feeding to the linear layer
        x = torch.cat((x, ext_input), dim=1)
        x = F.relu(self.fc1(x))
        pred_ext_input = F.relu(self.pred_ext_input(x))
        return x, pred_ext_input

    def decoder(self, x):
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, self.dim1, self.dim2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        return x

    def forward(self, x, ext_input):
        h, pred_ext_input = self.encoder(x, ext_input)
        out = self.decoder(h)
        return out, h, pred_ext_input

    def backward(self, optimizer, criterion, x, y_true, C_factor, alpha=0, ext_input=None):
        optimizer.zero_grad()

        y_pred, hidden, ext_input_pred = self.forward(x, ext_input)

        recon_loss = criterion(y_pred, y_true)
        recon_loss_ext_input = criterion(ext_input_pred, ext_input)

        # Whitening loss (batch whitening).
        hidden_constraint_loss = 0
        batch_size, hidden_dim = hidden.shape

        # SSCP matrix
        M = torch.mm(hidden.t(), hidden)

        # Covariance matrix
        I = torch.eye(hidden_dim, device='cuda')
        C = C_factor * I - M    # C = I - M    
        hidden_constraint_loss = alpha * torch.norm(C) / (batch_size * hidden_dim)
            
        loss = recon_loss + hidden_constraint_loss + recon_loss_ext_input #*1000
        loss.backward()

        optimizer.step()

        return recon_loss.item(), recon_loss_ext_input.item()



def create_dataloader(dataset, batch_size=256, reshuffle_after_epoch=True):
    if dataset.shape[-1] <= 3:
        dataset = np.transpose(dataset, (0,3,1,2))
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=reshuffle_after_epoch)


def predict(image, model, motiv_state=[]):
    output_ext_input = []
    if image.shape[-1] <= 4:
        image = np.transpose(image, (2, 0, 1))
    n_channels, n_pixels_height, n_pixels_width = image.shape
    image = np.reshape(image, (1, n_channels, n_pixels_height, n_pixels_width))
    image = torch.from_numpy(image).float().to(next(model.parameters()).device)

    if motiv_state != []:
        motiv_state = torch.Tensor(motiv_state[0]).float().unsqueeze(0).to(next(model.parameters()).device)
        output_img, _, output_ext_input = model(image, motiv_state)
        output_ext_input = output_ext_input[0].detach().cpu().numpy()
        output_img = output_img[0].detach().cpu().numpy()
    else:
        output_img = model(image)[0].detach().cpu().numpy()

    output_img = np.reshape(output_img, (n_channels, n_pixels_height, n_pixels_width))
    output_img = np.transpose(output_img, (1, 2, 0))

    return output_img, output_ext_input



def train_motivational_autoencoder(model, train_loader, external_inputs, C_factor, dataset=[], num_epochs=0, learning_rate=1e-4, alpha=2e3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model = model.to('cuda')
    external_inputs=torch.Tensor(external_inputs).to('cuda')

    history = []
    ext_history = []
    
    embeddings = []
    if len(dataset) > 0:
        embeddings = [get_latent_vectors(dataset=dataset, model=model)]

    for epoch in range(num_epochs):
        #print("Epoch =", epoch + 1)
        running_loss = 0.
        running_ext_loss = 0.
        for i, data in enumerate(train_loader, 0):            
            inputs, _ = data #image
            inputs = inputs.to('cuda')
            # Prepare external input for this batch
            ext_inputs_batch = external_inputs[i * train_loader.batch_size: (i + 1) * train_loader.batch_size]

            loss, ext_loss = model.backward(optimizer=optimizer, criterion=criterion, x=inputs, y_true=inputs, alpha=alpha, C_factor=C_factor, ext_input=ext_inputs_batch)
            running_loss += loss
            running_ext_loss += ext_loss

        history.append(running_loss / len(train_loader))
        ext_history.append(running_ext_loss / len(train_loader))

        if len(dataset) > 0:
            embeddings.append(get_latent_vectors(dataset=dataset, model=model))

    embeddings = np.array(embeddings)

    return history, embeddings, ext_history


def train_autoencoder(model, train_loader, C_factor, dataset=[], num_epochs=0, learning_rate=1e-4, alpha=2e3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model = model.to('cuda')

    history = []
    
    embeddings = []
    if len(dataset) > 0:
        embeddings = [get_latent_vectors(dataset=dataset, model=model)]

    for epoch in range(num_epochs):
        #print("Epoch =", epoch + 1)
        running_loss = 0.
        for i, data in enumerate(train_loader, 0):            
            inputs, _ = data #image
            inputs = inputs.to('cuda')

            loss = model.backward(optimizer=optimizer, criterion=criterion, x=inputs, y_true=inputs, alpha=alpha, C_factor=C_factor)
            running_loss += loss

        history.append(running_loss / len(train_loader))

        if len(dataset) > 0:
            embeddings.append(get_latent_vectors(dataset=dataset, model=model))

    embeddings = np.array(embeddings)

    return history, embeddings