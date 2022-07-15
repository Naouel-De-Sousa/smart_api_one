import sqlite3

import streamlit as st


conn = sqlite3.connect('data.db',check_same_thread=False)
c = conn.cursor()
conn.commit()

# Functions

def view_all_notes():
	
	html_temp = """
        <div style="background-color:#f02c0c;padding:10px;border-radius:10px">
        <h3 style="color:#FFFFFF;text-align:center;">Revolutionizing Our Hard Landing Response Times </h3>
		</div>"""
		
	st.write(html_temp.format('royalblue','white'),unsafe_allow_html=True)
	

# Login/Signup

def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data