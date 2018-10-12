import tkinter as tk
master = tk.Tk()
whatever_you_do = "Whatever you do will be insignificant, but it is very important that you do it.\n(Mahatma Gandhi)"
msg = tk.Message(master, text=whatever_you_do)
msg.config(bg='lightgreen', font=('times', 24, 'italic'),width=10000000)
msg.pack()
tk.mainloop()
