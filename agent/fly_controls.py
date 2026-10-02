"""Clickable fly controls, separate from MuJoCo's rendering shortcuts."""
import json
import tkinter as tk
from tkinter import ttk


class FlyControls:
    def __init__(self, send, channels, output):
        self.root=tk.Tk()
        self.root.title('Fly controls')
        self.root.geometry('440x470+40+80')
        self.root.attributes('-topmost',True)
        self.root.protocol('WM_DELETE_WINDOW',lambda:send(999))
        self.status=tk.StringVar(value='Starting fly…')
        ttk.Label(self.root,textvariable=self.status,wraplength=420).grid(row=0,column=0,columnspan=2,padx=10,pady=12)
        self.buttons={}
        def button(name, code, row, col):
            widget=ttk.Button(self.root,text=name,command=lambda:send(code))
            widget.grid(row=row,column=col,padx=8,pady=5,sticky='ew')
            self.buttons[name]=widget
        for i,channel in enumerate(channels):button(channel.replace('rival','visual').replace('_',' ').title(),49+i,1+i//2,i%2)
        for i,(name,code) in enumerate([('Pause / resume',32),('Sensory sweep',65),
                                      ('Reward',82),('Penalty',80),('Direct pulses on / off',68),
                                      ('Reset body / brain',78),('Vision on / off',86),('Stop fly',999)]):
            button(name,code,5+i//2,i%2)
        ttk.Label(self.root,text='Use these buttons. Keys in the 3D viewer belong to MuJoCo.',
                  wraplength=420).grid(row=9,column=0,columnspan=2,padx=10,pady=10)
        for col in range(2):self.root.columnconfigure(col,weight=1)
        self.root.update()
        rects={name:[b.winfo_rootx()+b.winfo_width()//2,b.winfo_rooty()+b.winfo_height()//2]
               for name,b in self.buttons.items()}
        (output/'control-buttons.json').write_text(json.dumps(rects,indent=2)+'\n')

    def update(self, text=None):
        if text:self.status.set(text)
        self.root.update()

    def close(self):
        self.root.destroy()
