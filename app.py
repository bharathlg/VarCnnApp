
import streamlit as st
import tensorflow as tf 
from tensorflow.keras.models import model_from_json
import PIL
import cv2

def get_model():
    ''' Returns Model loaded with original weights'''
    json_file = open('var600_model3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model=model_from_json(loaded_model_json)
    model.load_weights('weights.h5')
    return model

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# local_css('style.css')
st.markdown(
    """
    <style>
    .header-style {
        font-size:25px;
        font-family:sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title('VarCNN: The Automated Video Assistant Referee for Football')

st.markdown('<p class="header-style" >Football is the most followed sport in the world, played in more than 200M+ countries. The sport has developed a lot in the recent century and so has the technology involved in the game. The Virtual Assistant Referee (VAR) is one of them and has impacted the game to a large extent. The role of VAR is simple yet complex; to intervene in between the play when the referees make a wrong decision or cannot make one. A specific scenario arises when they have to decide if a sliding tackle inside the box has resulted in a clean tackle or penalty for the opposition team. The technology is there to watch the moment at which tackle took place on repeat but the decisions are still made by humans and hence can be biased. I propose a CNN based foul detection which is theoretically based on the principle of the initial point of contact.</p>',unsafe_allow_html=True)

image = PIL.Image.open('initial_contact.jpg')
st.image(image,caption='Initial Contact in a Tackle')


class app:
    def __init__(self):
        self.model=get_model()
    def sidebar(self):
        st.sidebar.markdown( """
        <style>
        .header-style {
        font-size:25px;
        font-family:sans-serif;
        }
        </style>
        """,unsafe_allow_html=True)
        st.sidebar.markdown('<p class="header-style">VAR In Action</p>',unsafe_allow_html=True)
    
        select=st.sidebar.selectbox('SELECT',['Foul','NoFoul'])

        if select=='Foul':
            video = open("foul-soccer.mp4", "rb") 
            st.sidebar.video(video)
            st.sidebar.write(f'You have slected, {select}')
            st.subheader('Simulation Initiated')
            st.markdown('<p class="header-style">Good choice!! Paulo Dybala, playing the ball and gets tackled!! We know it is one but can the VarCNN identify? Gear Up, We are going to know about that! </p>',unsafe_allow_html=True)
            
        else:
            video = open("CLEANTACKLE3.mp4", "rb") 
            st.sidebar.video(video)
            st.sidebar.write(f'You have slected, {select}')
            st.subheader('Simulation Initiated')
            st.markdown('<p class="header-style">Good choice!! A fine tackle by the player, gets the ball neat and clean!! We know it is one but can the VarCNN identify? Gear Up, We are going to know about that! </p>',unsafe_allow_html=True)
            
        return select

    def predict(self,select):
        pred=0
        print(select)
        st.sidebar.markdown( """
        <style>
        .foul {
        font-size:50px;
        font-family:sans-serif;
        color:red
        text-align:center;
        }
        </style>
        """,unsafe_allow_html=True)
        st.sidebar.markdown( """
        <style>
        .clean {
        font-size:50px;
        font-family:sans-serif;
        color:green
        text-align:center;
        }
        </style
        """,unsafe_allow_html=True)
        st.markdown('<p class="header-style">The predictions will be available soon</p>',unsafe_allow_html=True)

        if select=='Foul':
            vidcap = cv2.VideoCapture("foul-soccer.mp4")
        else:
            vidcap = cv2.VideoCapture("CLEANTACKLE3.mp4")
        success,image = vidcap.read()
        while success: 
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            if image is not None:
                print(image.shape)
                image=image/1./255
                image=image.astype('float32')
                image=cv2.resize(image,(256,256))
                print
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                print(image.shape)
                confidence=self.model.predict(image.reshape((1,256,256,3)))
                print('Confidence',confidence)
                if confidence[0][0]>=0.5:
                    pred=1
                    st.markdown('<p class="foul">Foul</p>',unsafe_allow_html=True)
                    if select=='Foul':
                        st.markdown('<p class="header-style">The model correctly identifies the foul :D</p>',unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="header-style">The model does not correctly identifies the foul :(</p>',unsafe_allow_html=True)
                    break
                       
        if not pred:
            # st.markdown('<p class="header-style">The predictions will be available soon</p>',unsafe_allow_html=True)
            st.markdown('<p class="clean">No Foul</p>',unsafe_allow_html=True)
            if select=='NoFoul':
                st.markdown('<p class="header-style">The model correctly identifies the clean tackle :D</p>',unsafe_allow_html=True)
            else:
                st.markdown('<p class="header-style">The model does not correctly identifies the clean tackle :(</p>',unsafe_allow_html=True)

          

    def construct(self):
        res=self.sidebar()
        self.predict(res)
        





run=app()

run.construct()
