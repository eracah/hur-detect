


def fit(model, generator, val_generator,callbacks, num_epochs):

    model.fit_generator(generator=generator,
                        steps_per_epoch= generator.num_ims / generator.batch_size,
                        validation_data=val_generator, 
                        validation_steps=val_generator.num_ims / val_generator.batch_size,
                        callbacks=callbacks,
                        epochs=num_epochs,
                        workers=4
                       )





