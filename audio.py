from gtts import gTTS
import argparse,os
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
    default="hello",
    help="path to trained face mask detector model")
args = vars(ap.parse_args())
#def audio_out():
language = 'en'
myobj = gTTS(text=args["model"], lang=language, slow=False) 
myobj.save("welcome1.mp3") 
      
    # Playing the converted file 
os.system("mpg321 welcome1.mp3")
#audio_out()