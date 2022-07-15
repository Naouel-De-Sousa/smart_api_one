import streamlit as st
import os

# page title
from PIL import Image
img = Image.open("./Logo/logo.png")
st.set_page_config(page_title = "smart lander api", page_icon = img)


# EDA Pkgs
import pandas as pd 

# DB
from db_fix import *
# prediction
import pickle,joblib
import csv

# Security
import hashlib

def make_hashes(passwd):
    return hashlib.sha256(str.encode(passwd)).hexdigest()

def check_hashes(passwd,hashed):
    if make_hashes(passwd) == hashed:
        return hashed
    return False



# Layout Templates
title_temp ="""
    <div style="background-color:#f02c0c;padding:10px;border-radius:10px;margin:10px;">
    <h4 style="color:white;text-align:center;">{}</h1>
    <img src="/Logo/logo.png" alt="ATR logo"/ style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
    <h6>predict_TZ_target:{}</h6>
    <br/>
    <br/>	
    <p style="text-align:justify">{}</p>
    </div>
    """

full_message_temp ="""
    <div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
        <p style="text-align:justify;color:black;padding:10px">{}</p>
    </div>
    """



    
# center logo ATR
@st.cache
def load_image(img):
	im =Image.open(os.path.join(img))
	return im
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image(load_image("logo/Atr_new.png"))

with col3:
    st.write(' ')


# main function
def main():
    """A simple application"""
    html_temp = """
        <div style="background-color:#f02c0c;padding:10px;border-radius:10px">
        <h1 style="color:#FFFFFF;text-align:center;">Smart Lander </h1>
        <h2 style="color:#FFFFFF;text-align:center;">Application to predict the vertical forces undergone by a landing gear </h2>  </div>
        """
    st.markdown(html_temp.format('royalblue','white'),unsafe_allow_html=True)

    menu = ["Home","Login","SignUp"]
    #sub_menu = ["Left vertical ground force","Right vertical ground force","Users Profile"]

    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home")
        result = view_all_notes()



## login session

    elif choice == "Login":
        username = st.sidebar.text_input('Username')
        passwd = st.sidebar.text_input('Password',type='password')
        if st.sidebar.checkbox('Login') :
            create_usertable()
            hashed_pswd = make_hashes(passwd)
            result = login_user(username,check_hashes(passwd,hashed_pswd))
            if result:
                st.success("Logged In as {}".format(username))
                #task = st.selectbox("Activity",sub_menu)
                #task = st.selectbox('Select Task',["Left vertical ground force","Right vertical ground force","Users Profile"])
                #task = st.multiselect("Select a Task: ",["predict vertical ground force on air","predict verticalg round force on oil","Users Profile"])
                #task = st.checkbox("Navigation",tasks)
                task1 = st.checkbox("Left vertical ground force")
                task2 = st.checkbox("Right vertical ground force")
                task3 = st.checkbox("Users Profile")



#prediction for first model
        
    # input user
                if task1: #"Left vertical ground force":
                    st.subheader("Enter your flight data")
                    st.write("""The user have to enter the features to have prediction
                                for verical ground force""")

    # function to user input
                    def user_input_features():
                        masse = st.number_input('masse',17500.0,22350.0,20000.0)
                        NZ1 = st.number_input('NZ1',0.00000,3.78640,0.00100,)
                        PITCH_1 = st.number_input('PITCH_1',-6.3100,6.0000,0.00)
                        ROLL_1 = st.number_input('ROLL_1', -5.7700,5.8800,1.0000)
                        Vx_GRD = st.number_input('Vx_GRD',0.0000,61.7328,6.0000)
                        Delta_Ny_AC = st.number_input('Delta_Ny_AC', -0.7694,0.3592,0.0000)
                        Wy_AC = st.number_input('Wy_AC', -3.0000,8.0000,0.0000)
                        Wx_AC = st.number_input('Wx_AC',-10.00,10.00,5.00)
                        time_nz_max = st.number_input('time_nz_max', 0.0000,0.9950,0.0100,)
                        delta_pitch = st.number_input('delta_pitch', -1.0200,0.6100,0.0000)
                        delta_roll = st.number_input('delta_roll', -1.6700,1.8400,0.0000)
                        delta_nz = st.number_input('delta_nz', 0.0000,3.8205,0.5000)

                        # return build_dict_convert_df(masse = masse,NZ1 = NZ1 ,PITCH_1 =PITCH_1 ,
                        #     ROLL_1=ROLL_1 , 
                        #     Vx_GRD = Vx_GRD,
                        #     Delta_Ny_AC = Delta_Ny_AC,
                        #     Wy_AC = Wy_AC,
                        #     Wx_AC =Wx_AC,
                        #     time_nz_max =time_nz_max,
                        #     delta_pitch = delta_pitch,
                        #     delta_roll =delta_roll ,
                        #     delta_nz =delta_nz)
                            
                    #input_df = user_input_features()   
                        data = { 'masse':[masse],
                                     'NZ1':[NZ1], 
                                     'PITCH_1':[PITCH_1], 
                                     'ROLL_1':[ROLL_1], 
                                     'Vx_GRD':[Vx_GRD],
                                     'Delta_Ny_AC':[Delta_Ny_AC],
                                     'Wy_AC':[Wy_AC],
                                     'Wx_AC':[Wx_AC],
                                     'time_nz_max':[time_nz_max],
                                     'delta_pitch':[delta_pitch],
                                     'delta_roll':[delta_roll],
                                     'delta_nz':[delta_nz]
                                     }
                        features = pd.DataFrame(data)
                        return features

                    input_df = user_input_features()



        # visualize user input

                    st.markdown('<h3>Check the flight data</h3>', unsafe_allow_html=True)
                    st.write(input_df)
                    st.title("Left vertical ground force")
                    scaler_TZ_AC_MLGL_air_max= joblib.load(open("./scalers/scaler_TZ_AC_MLGL_air_max.save",'rb'))
                    df = pd.DataFrame(scaler_TZ_AC_MLGL_air_max.transform(input_df), columns = input_df.columns)
       
       
        ## load the model file
                    load_model_TZ_AC_MLGL_air_max = pickle.load(open('./models/model_TZ_AC_MLGL_air_max.pkl', 'rb'))
                
            # use the model to predict target
                    if st.button('predict Left vertical ground force "air"'):
                        prediction = load_model_TZ_AC_MLGL_air_max.predict(df)
                        st.success('Vertical ground force "air" for main landing gear left is {} daN'.format(prediction))

            # convert result in csv           
                        prediction_MLGL = float(prediction)
                        colonnes = ['TZ_AC_MLGL_air_max']

                # convert prediction to dataframe
                        prediction_MLGL = pd.DataFrame(data = [prediction_MLGL], columns = colonnes )
                        
                 # concat choosen features and prediction to create dataframe
                        result_mlgl = pd.concat([input_df,prediction_MLGL], axis = 1)
                        
                # convert result to csv and save it
                        def convert_df_to_csv(result_mlgl):
                            return result_mlgl.to_csv().encode('utf-8')    
                        st.download_button(label="Download data as CSV",data = convert_df_to_csv(result_mlgl),
                                                file_name='prediction_TZ_MLGL.csv',mime='text/csv') 


                


                elif task2: #== "Right vertical ground force": 
                    
                    st.subheader("Right vertical ground force")
                    def user_input_features():
                        masse = st.number_input('masse',17500.0,22350.0,20000.0)
                        NZ1 = st.number_input('NZ1',0.00000,3.78640,0.00100,)
                        PITCH_1 = st.number_input('PITCH_1',-6.3100,6.0000,0.00)
                        ROLL_1 = st.number_input('ROLL_1', -5.7700,5.8800,1.0000)
                        Vx_GRD = st.number_input('Vx_GRD',0.0000,61.7328,6.0000)
                        Delta_Ny_AC = st.number_input('Delta_Ny_AC', -0.7694,0.3592,0.0000)
                        Wy_AC = st.number_input('Wy_AC', -3.0000,8.0000,0.0000)
                        Wx_AC = st.number_input('Wx_AC',-10.00,10.00,5.00)
                        time_nz_max = st.number_input('time_nz_max', 0.0000,0.9950,0.0100,)
                        delta_pitch = st.number_input('delta_pitch', -1.0200,0.6100,0.0000)
                        delta_roll = st.number_input('delta_roll', -1.6700,1.8400,0.0000)
                        delta_nz = st.number_input('delta_nz', 0.0000,3.8205,0.5000)


                        data = { 'masse':[masse],
                                     'NZ1':[NZ1], 
                                     'PITCH_1':[PITCH_1], 
                                     'ROLL_1':[ROLL_1], 
                                     'Vx_GRD':[Vx_GRD],
                                     'Delta_Ny_AC':[Delta_Ny_AC],
                                     'Wy_AC':[Wy_AC],
                                     'Wx_AC':[Wx_AC],
                                     'time_nz_max':[time_nz_max],
                                     'delta_pitch':[delta_pitch],
                                     'delta_roll':[delta_roll],
                                     'delta_nz':[delta_nz]
                                     }
                        features = pd.DataFrame(data)
                        return features


              
                            
                    input_df = user_input_features()  
 
                    scaler_TZ_AC_MLGR_air_max= joblib.load(open("./scalers/scaler_TZ_AC_MLGR_air_max.save",'rb'))
                    df = pd.DataFrame(scaler_TZ_AC_MLGR_air_max.transform(input_df), columns = input_df.columns)

            # load the model file
                    load_model_TZ_AC_MLGR_air_max = pickle.load(open('./models/model_TZ_AC_MLGR_air_max.pkl', 'rb'))

            # use the model to predict target
                    if st.button('predict Right vertical ground force "air" '):
                        prediction = load_model_TZ_AC_MLGR_air_max.predict(df)
                        st.success('Vertical ground force "air" of main lainding gear right is {} daN'.format(prediction))

            # Download result to csv file
                        prediction_MLGR = float(prediction)
                        colonnes = ["TZ_AC_MLGR_air_max"]
                        prediction_MLGR = pd.DataFrame(data = [[prediction_MLGR]], columns = colonnes )

            # concat choosen features and prediction to create dataframe
                        result_mlgr = pd.concat([input_df,prediction_MLGR], axis = 1)
            # convert result to csv and save it
                        def convert_df_to_csv(result_mlgr):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                            return result_mlgr.to_csv().encode('utf-8')    
                        st.download_button(label="Download data as CSV",data = convert_df_to_csv(result_mlgr),
                                                file_name='prediction_TZ_MLGR.csv',mime='text/csv')

                    
                   
        # show user profile   


                elif task3: #== "Users Profile":
                    user_result = view_all_users()
                    clean_db = pd.DataFrame(user_result,columns=['User','Password'])
                    st.dataframe(clean_db)                                   

            else:
                st.warning("Incorrect Username and Password")
                st.info('Retry or SignUp')


        #sign up session


    elif choice == "SignUp":
        st.subheader("Create an Account")
        new_user = st.text_input('Username')
        new_passwd = st.text_input('Password',type='password')
        if st.button('SignUp'):
            create_usertable()
            add_userdata(new_user,make_hashes(new_passwd))
            st.success("You have successfully created an account.Go to the Login Menu to login") 


     # Tasks For Only Logged In Users
        







if __name__ == '__main__':
    main()