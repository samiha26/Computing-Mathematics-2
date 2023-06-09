import streamlit as st
import numpy as np
import pandas as pd
from cmfrec import CMF
from PIL import Image

# Load the ratings data
ratings = pd.read_csv("C:\\Users\\hp\\OneDrive\\Desktop\\Apple_Pi\\df_rearranged.csv")

# Load the app names data
app_names = pd.read_csv("C:\\Users\\hp\\OneDrive\\Desktop\\Apple_Pi\\df_app_names.csv")

# Load the collaborative filtering model
model = CMF(method="als", k=26, lambda_=1e+1)


# Define the Streamlit app
def main():
    # Set the page title
    st.title("Student Productivity App Recommendation System (Rate the Apps you already used)")

    # Retrieve the user ratings for each app
    user_ratings = []  # Store the user ratings here

    # Display the app pictures and rating stars
    for app_id in range(26):
        # Print the app name
        app_name = app_names.AppName.loc[app_names.AppID == app_id+1].values[0]
        st.write(f"- {app_name}")
        
        # Display the app picture
        st.image(Image.open(f"C:\\Users\\hp\\OneDrive\\Desktop\\Apple_Pi\\app_images\\app{app_id + 1}.png"), width=100)

        # Display the rating stars
        rating = st.slider("Rate the app", 0, 5, key=app_id+1)
        

        # Store the user rating for the app
        user_ratings.append(rating)
        

    if st.button("Generate Recommendations"):
        # Making app recommendations based on user ratings


        #We append the new user with userId 10000 and the apps he rated with thier respective ratings
        for i in range(len(user_ratings)):
            if user_ratings[i]>0:
                new_user_rating = {'UserId':10000, 'ItemId':i+1,'Rating':user_ratings[i] }
                ratings.loc[len(ratings)] = new_user_rating
        
        #After the new user has been added we train the collaborative filtering model with the users data
        model.fit(ratings)

        #And then we check for recomendations for the user with UserId = 10000 (meaning the new user :)
        exclude = ratings.ItemId.loc[ratings.UserId == 10000] #excluding already rated apps :p
        recommended_apps = model.topN(user = 10000, n=5, exclude = exclude)

        # Display the recommended app pictures
        st.subheader("Recommended Apps")
        for app_id in recommended_apps:
            print(app_id)
            app_name = app_names.AppName.loc[app_names.AppID == app_id].values[0]
            st.write(f"- {app_name}")
            st.image(Image.open(f"C:\\Users\\hp\\OneDrive\\Desktop\\Apple_Pi\\app_images\\app{app_id}.png"), width=100)

if __name__ == "__main__":
    main()
