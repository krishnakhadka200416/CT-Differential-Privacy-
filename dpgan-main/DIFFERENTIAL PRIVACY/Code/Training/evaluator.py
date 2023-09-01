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

