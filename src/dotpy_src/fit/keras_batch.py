


def fit(model, generator, val_generator,callbacks, num_epochs):
    steps_per_epoch= generator.num_ims / generator.batch_size
    validation_steps=val_generator.num_ims / val_generator.batch_size
    
    for ep in range(num_epochs):
        for _ in range(steps_per_epoch):
            ims,lbls = generator.next()
            model.train_on_batch(x=ims,y=lbls)
        for _ in range(validation_steps):
            val_ims, val_lbls = val_generator.next()
            print "validation: ", model.evaluate(val_ims, val_labels)





