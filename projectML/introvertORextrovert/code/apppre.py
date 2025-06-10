import tkinter as tk
from tkinter import messagebox
import joblib

# Load model
path = r'C:\Users\97ngu\OneDrive\Desktop\course\ML\projectML\introvertORextrovert\model\logistic_model.pkl'
loaded_model = joblib.load(path)

# Hàm xử lý dự đoán
def predict_personality():
    try:
        Time_spent_Alone = int(entry_time_alone.get())
        Stage_fear = 1 if entry_stage_fear.get().strip().lower() == 'yes' else 0
        Social_event_attendance = int(entry_social_event.get())
        Going_outside = int(entry_going_outside.get())
        Drained_after_socializing = 1 if entry_drained.get().strip().lower() == 'yes' else 0
        Friends_circle_size = int(entry_friends_circle.get())
        Post_frequency = int(entry_post_freq.get())

        user_input = [[Time_spent_Alone, Stage_fear, Social_event_attendance, Going_outside,
                       Drained_after_socializing, Friends_circle_size, Post_frequency]]

        prediction = loaded_model.predict(user_input)

        result = "Introvert" if prediction == [1] else "Extrovert"
        messagebox.showinfo("prediction result", f"you are: {result}")

    except ValueError:
        messagebox.showerror("error", "please enter again exactly input!")

# Hàm reset dữ liệu
def reset_fields():
    for entry in entries:
        entry.delete(0, tk.END)

# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("What are you type ? ")

# Tạo các label và entry
labels = ["Time spent Alone (day):", "Stage fear (yes/no):", "Social event attendance (times/month):",
          "Going outside (hours):", "Drained after socializing (yes/no):", 
          "Friends circle size (amount):", "Post frequency (times/month):"]

entries = []

for i, label_text in enumerate(labels):
    tk.Label(root, text=label_text).grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entries.append(entry)

# Lưu các entry vào biến để dễ truy cập
(entry_time_alone, entry_stage_fear, entry_social_event,
 entry_going_outside, entry_drained, entry_friends_circle, entry_post_freq) = entries

# Nút dự đoán
tk.Button(root, text="Prediction", command=predict_personality).grid(row=len(labels), column=0)

# Nút Reset
tk.Button(root, text="Reset", command=reset_fields).grid(row=len(labels), column=1)

# Chạy ứng dụng
root.mainloop()