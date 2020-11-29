import gpt_2_simple as gpt2
import os

model_name = "355M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)

name_of_artist = 'travis'

file_name = f"training_data/{name_of_artist}.txt"

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='355M',
              steps=1000,
              restore_from='fresh',
              run_name=f'{name_of_artist}',
              print_every=1,
              sample_every=200,
              save_every=200
              )

gpt2.generate(sess,
              length=500,
              temperature=0.8,
              prefix="It's lit",
              nsamples=25,
              batch_size=5
              )
