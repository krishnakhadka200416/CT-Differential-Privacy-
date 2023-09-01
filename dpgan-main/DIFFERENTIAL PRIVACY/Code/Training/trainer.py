def apply_differential_privacy(discriminator, train_dataloader):
    DELTA = 1e-5
    criterion = nn.BCELoss()
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)



    privacy_engine=PrivacyEngine()

    discriminator, optimizer_D, train_dataloader = privacy_engine.make_private(
    module=discriminator,
    optimizer=optimizer_D,
    data_loader=train_dataloader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)



    for epoch in tqdm(range(50), desc="Training Epochs"):
        losses=[]
        for data, _ in train_dataloader:
            data=data.to(device)
            batch_size = data.shape[0]
            #print(f' before data {data.shape}')
            data = data.view(batch_size, -1)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1,dtype=torch.float32)
            fake_labels = torch.zeros(batch_size, 1,dtype=torch.float32)

            real_labels=real_labels.to(device)
            fake_labels=fake_labels.to(device)

            real_output = discriminator(data)
          
            real_loss = criterion(real_output, real_labels)
            losses.append(real_loss.item())

            z = torch.randn(batch_size, 6000,device=device,dtype=torch.float32)
            
            fake_data = generator(z)
            fake_output = discriminator(fake_data.detach())

            
            fake_loss = criterion(fake_output, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

        epsilon = privacy_engine.accountant.get_epsilon(delta=DELTA)
        tqdm.write(
            f"Train Epoch: {epoch} \t"
            f"Loss: {numpy.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {DELTA})"
        )
