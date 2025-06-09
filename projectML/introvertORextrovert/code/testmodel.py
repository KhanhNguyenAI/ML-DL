import joblib
path = r'C:\Users\97ngu\OneDrive\Desktop\course\ML\projectML\introvertORextrovert\model\logistic_model.pkl'
loaded_model = joblib.load(path)
Time_spent_Alone = input('Time spent Alone (day): ')
Stage_fear = input('Stage fear(yes/no): ').strip().lower()
Social_event_attendance = input('Social event attendance ( times/moth): ')
Going_outside = input('Going outside (hours): ')
Drained_after_socializing = input('Drained after socializing (yes/no): ').strip().lower()
Friends_circle_size = input('Friends circle size (amount): ')
Post_frequency = input('Post frequency (times/moth): ')

if Stage_fear == 'yes' : 
    Stage_fear_l = 1
elif Stage_fear == 'no' : 
    Stage_fear_l = 0
else : 
    Stage_fear_l = Stage_fear
if Drained_after_socializing == 'yes' : 
    Drained_after_socializing_l = 1
elif Drained_after_socializing == 'no' :
    Drained_after_socializing_l = 0
else : 
    Drained_after_socializing_l = Drained_after_socializing

user_input = [[int(Time_spent_Alone),int(Stage_fear_l),int(Social_event_attendance),int(Going_outside),int(Drained_after_socializing_l),
               int(Friends_circle_size),int(Post_frequency)]]
print(type(user_input))
print(user_input)
# [[11, 1, 3, 0, 1, 5, 0]] 

model = loaded_model.predict(user_input)
if model == [1] : 
    print('Introvert')
else :
    print('Extrovert')
