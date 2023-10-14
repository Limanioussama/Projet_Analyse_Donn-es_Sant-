from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from imblearn.over_sampling import SMOTE

# Read the CSV file into a pandas DataFrame

df = pd.read_csv('brain_stroke.csv', delimiter=';')


# Define the columns to label encode
columns_to_encode = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Using label encoder to labelling the object_type data

# Create an instance of LabelEncoder for each column
label_encoders = {}

for column in columns_to_encode:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Print the encoded labels


# Save the updated DataFrame to a new CSV file
df.to_csv('encoded_file2.csv', index=False)
new_df = pd.read_csv('encoded_file2.csv', delimiter=';')
print("new_df")
print(new_df.head())

# Split the data into train and test:
X = df.drop(columns=['stroke'])
y = df['stroke']
s1=SMOTE()

x_data,y_data=s1.fit_resample(X,y)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state = 37)

print(x_train.shape, x_test.shape)

knn = KNeighborsClassifier(n_neighbors=3)  # Utilisation de k=3 pour les k-plus proches voisins
knn.fit(x_train, y_train)

# Étape 4 : Évaluer les performances du modèle sur l'ensemble de test
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print('Précision du modèle :', accuracy)

# Étape 5 : Prédire la classe d'une nouvelle fleur iris
new_stroke = [[1,75.0,0,0,1,0,1,94.29,35.2,0]]
predicted_class = knn.predict(new_stroke)
print('la prediction du stroke pour cette echantillion est :', predicted_class)



# Python program to create a basic form
# GUI application using the customtkinter module

import tkinter.messagebox as tkmb
from PIL import ImageTk
from PIL import Image
import customtkinter as ctk
import tkinter as tk

# Dimensions of the window
appWidth, appHeight = 950, 800

#,avg_glucose_level,bmi,smoking_status
# App Class
class App(ctk.CTk):
    # The layout of the window will be written
    # in the init function itself
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Sets the title of the window to "App"
        self.title("Stroke prediction")
        # Sets the dimensions of the window to 600x700
        self.geometry(f"{appWidth}x{appHeight}")

        # Name Label
        self.nameLabel = ctk.CTkLabel(self,
                                      text="Name")
        self.nameLabel.grid(row=0, column=0,
                            padx=15, pady=15,
                            sticky="ew")

        # Name Entry Field
        self.nameEntry = ctk.CTkEntry(self,
                                      placeholder_text="name")
        self.nameEntry.grid(row=0, column=1,
                            columnspan=3, padx=15,
                            pady=15, sticky="ew")

        self.resLabel = ctk.CTkLabel(self,
                                    text="Generated results",
                                    corner_radius=8,
                                    font=('Calibri', 30))
        self.resLabel.grid(row=0, column=4,
                            columnspan=3, rowspan=1,
                            padx=15, pady=15,
                            sticky="ew")

        # Age Label
        self.ageLabel = ctk.CTkLabel(self, text="Age")
        self.ageLabel.grid(row=1, column=0,
                           padx=15, pady=15,
                           sticky="ew")

        # Age Entry Field
        self.ageEntry = ctk.CTkEntry(self,
                                     placeholder_text="18")
        self.ageEntry.grid(row=1, column=1,
                           columnspan=3, padx=15,
                           pady=15, sticky="ew")
        #deisplay board
        self.displayBox = ctk.CTkTextbox(self, height=200, width=300)
        self.displayBox.grid(row=1, column=4,
                             columnspan=4, rowspan=4, padx=15,
                             pady=15, sticky="nsew")
        # Gender Label
        self.genderLabel = ctk.CTkLabel(self,
                                        text="Gender")
        self.genderLabel.grid(row=2, column=0,
                              padx=15, pady=15,
                              sticky="ew")

        # Gender Radio Buttons
        self.genderVar = tk.StringVar(value="")

        self.maleRadioButton = ctk.CTkRadioButton(self,
                                                  text="Male",
                                                  variable=self.genderVar,
                                                  value="1")
        self.maleRadioButton.grid(row=2, column=1,
                                  padx=15, pady=15,
                                  sticky="ew")

        self.femaleRadioButton = ctk.CTkRadioButton(self,
                                                    text="Female",
                                                    variable=self.genderVar,
                                                    value="0")
        self.femaleRadioButton.grid(row=2, column=2,
                                    padx=15, pady=15,
                                    sticky="ew")

        # hypertension Label
        self.hypertensionLabel = ctk.CTkLabel(self,
                                        text="hypertension")
        self.hypertensionLabel.grid(row=3, column=0,
                              padx=15, pady=15,
                              sticky="ew")

        # hypertension radioChoice

        self.hypertensionVar = tk.StringVar(value="hypertension")

        self.hyesRadioButton = ctk.CTkRadioButton(self,
                                              text="Yes",
                                              variable=self.hypertensionVar,
                                              value="1")
        self.hyesRadioButton.grid(row=3, column=1,
                                padx=15, pady=15,
                                 sticky="ew")

        self.hnoRadioButton = ctk.CTkRadioButton(self,
                                             text="No",
                                             variable=self.hypertensionVar,
                                             value="0")
        self.hnoRadioButton.grid(row=3, column=2,
                             padx=15, pady=15,
                             sticky="ew")
        # Choice Label heart_d

        self.hdLabel = ctk.CTkLabel(self,
                                        text="Heart disease")
        self.hdLabel.grid(row=4, column=0,
                              padx=15, pady=15,
                              sticky="ew")
        self.hdVar = tk.StringVar(value="heart_disease")

        self.hdyesRadioButton = ctk.CTkRadioButton(self,
                                                   text="Yes",
                                                   variable=self.hdVar,
                                                   value="1")
        self.hdyesRadioButton.grid(row=4, column=1,
                                   padx=15, pady=15,
                                   sticky="ew")

        self.hdnoRadioButton = ctk.CTkRadioButton(self,
                                                  text="No",
                                                  variable=self.hdVar,
                                                  value="0")
        self.hdnoRadioButton.grid(row=4, column=2,
                                  padx=15, pady=15,
                                  sticky="ew")
        # Avg Label
        self.avgLabel = ctk.CTkLabel(self, text="average glucose level")
        self.avgLabel.grid(row=5, column=0,
                           padx=15, pady=15,
                           sticky="ew")

        # Avg Entry Field
        self.avgEntry = ctk.CTkEntry(self,
                                     placeholder_text="150")
        self.avgEntry.grid(row=5, column=1,
                           columnspan=3, padx=15,
                           pady=15, sticky="ew")

        # bmi Label
        self.bmiLabel = ctk.CTkLabel(self, text="Poid/taille")
        self.bmiLabel.grid(row=6, column=0,
                           padx=15, pady=15,
                           sticky="ew")

        # bmi Entry Field
        self.bmiEntry = ctk.CTkEntry(self,
                                     placeholder_text="45")
        self.bmiEntry.grid(row=6, column=1,
                           columnspan=3, padx=15,
                           pady=15, sticky="ew")

        # ever_married Label
        self.marrieLabel = ctk.CTkLabel(self,
                                        text="Ever married")
        self.marrieLabel.grid(row=7, column=0,
                              padx=15, pady=15,
                              sticky="ew")

        # ever_maried Radio Buttons
        self.marrieVar = tk.StringVar(value="Prefer")

        self.marriedRadioButton = ctk.CTkRadioButton(self,
                                                  text="Married",
                                                  variable=self.marrieVar,
                                                  value="1")
        self.marriedRadioButton.grid(row=7, column=1,
                                  padx=15, pady=15,
                                  sticky="ew")

        self.notmarriedRadioButton = ctk.CTkRadioButton(self,
                                                    text="Not married",
                                                    variable=self.marrieVar,
                                                    value="0")
        self.notmarriedRadioButton.grid(row=7, column=2,
                                    padx=15, pady=15,
                                    sticky="ew")
        # smoke Label
        self.smokeLabel = ctk.CTkLabel(self,
                                       text="Smoking")
        self.smokeLabel.grid(row=8, column=0,
                              padx=15, pady=15,
                              sticky="ew")
        # Unknown0smokes3never smoked2formerly smoked1
        # Gender Radio Buttons
        self.smokeVar = tk.StringVar(value="Unknown")

        self.unknownRadioButton = ctk.CTkRadioButton(self,
                                                     text="Unknown",
                                                     variable=self.smokeVar,
                                                     value="0")
        self.unknownRadioButton.grid(row=8, column=1,
                                     padx=8, pady=8,
                                     sticky="ew")

        self.formerlyRadioButton = ctk.CTkRadioButton(self,
                                                      text="formerly smoked",
                                                      variable=self.smokeVar,
                                                      value="1")
        self.formerlyRadioButton.grid(row=8, column=2,
                                      padx=8, pady=8,
                                      sticky="ew")

        self.neverRadioButton = ctk.CTkRadioButton(self,
                                                   text="never smoked",
                                                   variable=self.smokeVar,
                                                   value="2")

        self.neverRadioButton.grid(row=8, column=3, padx=8,
                                   pady=8, sticky="ew")
        self.smokesRadioButton = ctk.CTkRadioButton(self,
                                                    text="smokes",
                                                    variable=self.smokeVar,
                                                    value="3")
        self.smokesRadioButton.grid(row=8, column=4, padx=8,
                                    pady=8, sticky="ew")

        # Occupation Label
        self.occupationLabel = ctk.CTkLabel(self,
                                            text="Work type")
        self.occupationLabel.grid(row=9, column=0,
                                  padx=15, pady=15,
                                  sticky="ew")

        # Occupation combo box
        self.occupationOptionMenu = ctk.CTkOptionMenu(self,
                                                      values=["Private",
                                                              "Self-employed",
                                                              "Govt_job",
                                                              "children"])
        self.occupationOptionMenu.grid(row=9, column=1,
                                       padx=15, pady=15,
                                       columnspan=2, sticky="ew")

        # Residence_type Label
        self.residenceLabel = ctk.CTkLabel(self,
                                           text="Residence")
        self.residenceLabel.grid(row=10, column=0,
                                 padx=15, pady=15,
                                 sticky="ew")

        # Residence Radio Buttons
        self.residenceVar = tk.StringVar(value="0")

        self.urbanRadioButton = ctk.CTkRadioButton(self,
                                                   text="urban",
                                                   variable=self.residenceVar,
                                                   value="0")
        self.urbanRadioButton.grid(row=10, column=1,
                                   padx=15, pady=15,
                                   sticky="ew")

        self.ruralRadioButton = ctk.CTkRadioButton(self,
                                                   text="rural",
                                                   variable=self.residenceVar,
                                                   value="1")
        self.ruralRadioButton.grid(row=10, column=2,
                                   padx=15, pady=15,
                                   sticky="ew")

        # Generate Button
        self.generateResultsButton = ctk.CTkButton(self,
                                                   text="Generate Results",
                                                   command=self.generateResults)
        self.generateResultsButton.grid(row=11, column=2,
                                        columnspan=3, padx=15,
                                        pady=15, sticky="ew")



    def generateResults(self):
        self.displayBox.delete("0.0", "200.0")
        text = self.createText()
        self.displayBox.insert("0.0", text)


    def createText(self):

        # Constructing the text variable
        #text = f"{self.nameEntry.get()} : \n{self.genderVar.get()} {self.ageEntry.get()} years old and prefers \n"
        #text += f"{self.genderVar.get()} currently a {self.occupationOptionMenu.get()}"

        text = f"{self.nameEntry.get()} : {self.ageEntry.get()} years old \n"
        # text += f"{self.genderVar.get()} currently a {self.occupationOptionMenu.get()}"

        # gender
        gender = int(self.genderVar.get())
        # age
        age = int(self.ageEntry.get())
        # hypertension
        hypertension = int(self.hypertensionVar.get())
        # heart_disease
        heart_disease = int(self.hdVar.get())
        # never_married
        ever_married = int(self.marrieVar.get())
        # work_type

        #["Private"  "Self-employed",  0"Govt_job", "children"]
        if self.occupationOptionMenu.get() == "Private":
            work_type = 1
        elif self.occupationOptionMenu.get() == "Self-employed":
            work_type = 2
        elif self.occupationOptionMenu.get() == "Govt_job":
            work_type = 0
        else :
            work_type = 3

        # Residence_type
        Residence_type = int(self.residenceVar.get())
        # avg_glucose_level
        avg_glucose_level = int(self.avgEntry.get())
        # bmi
        bmi = int(self.bmiEntry.get())
        # smoking_status
        smoking_status = int(self.smokeVar.get())

        new_stroke = [
            [gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi,
             smoking_status]]
        predicted_class = knn.predict(new_stroke)
        if predicted_class==[0]:
            pred_class="negative"
        else :
            pred_class = "positive"
        # print('la prediction du stroke pour cette echantillion est :', predicted_class)

        text += f"\n the stroke prediction for this person is :{pred_class}"
        return text

def login():
    username = "hamza"
    password = "12345"
    if user_entry.get() == username and user_pass.get() == password:
        tkmb.showinfo(title="Login Successful", message="You have logged in Successfully")
        gui = App()
        app.destroy()
        gui.mainloop()
    elif user_entry.get() == username and user_pass.get() != password:
        tkmb.showwarning(title='Wrong password', message='Please check your password')
    elif user_entry.get() != username and user_pass.get() == password:
        tkmb.showwarning(title='Wrong username', message='Please check your username')
    else:
        tkmb.showerror(title="Login Failed", message="Invalid Username and password")

if __name__ == "__main__":
    # Sets the appearance of the window
    # Supported modes : Light, Dark, System
    # "System" sets the appearance mode to
    # the appearance mode of the system
    ctk.set_appearance_mode("Dark")


    # Selecting color theme - blue, green, dark-blue
    ctk.set_default_color_theme("dark-blue")

    app = ctk.CTk()
    app.geometry("500x650")
    app.title("Stroke prediction")

    label = ctk.CTkLabel(app,
                         text="Stroke prediction helper for doctors\n using machine learning",
                         width=120,
                         height=25,
                         corner_radius=8,
                         font=('Calibri', 30))
    label.pack(pady=20)

    image = ctk.CTkImage(Image.open("stroke.jpg"), size=(220, 200))
    logoLabel = ctk.CTkLabel(app, image=image, text='')
    logoLabel.pack(pady=12, padx=10)

    frame = ctk.CTkFrame(master=app)
    frame.pack(pady=20, padx=40, fill='both', expand=True)

    label = ctk.CTkLabel(master=frame, text='Login for doctors')
    label.pack(pady=12, padx=10)

    user_entry = ctk.CTkEntry(master=frame, placeholder_text="Username")
    user_entry.pack(pady=12, padx=10)

    user_pass = ctk.CTkEntry(master=frame, placeholder_text="Password", show="*")
    user_pass.pack(pady=12, padx=10)

    button = ctk.CTkButton(master=frame, text='Login', command=login)
    button.pack(pady=12, padx=10)

    checkbox = ctk.CTkCheckBox(master=frame, text='Remember Me')
    checkbox.pack(pady=12, padx=10)

    # Used to run the application
    app.mainloop()
