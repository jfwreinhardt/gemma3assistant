import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM, TextIteratorStreamer
from threading import Thread
import tkinter as tk
from tkinter import ttk
import queue

# Copyright 2025 JFW Reinhardt
# https://github.com/jfwreinhardt/gemma3assistant
# Apache 2.0 License
# http://www.apache.org/licenses/LICENSE-2.0

class GemmaUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gemma 3 Local AI Assistant")
        self.root.geometry("720x600")

        # Initialize model and tokenizer
        ckpt = r"C:\gemma3_1b_it_model"
        self.model = Gemma3ForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)

        self.setup_ui()
        self.response_queue = queue.Queue()

    def setup_ui(self):
        # Expertise input
        expertise_frame = ttk.Frame(self.root)
        expertise_frame.pack(pady=10, padx=10, fill='x')

        ttk.Label(expertise_frame,
                  text="In just a word or two, describe what you need the assistant to be an expert in:").pack(fill='x')
        self.expertise_entry = ttk.Entry(expertise_frame)
        self.expertise_entry.pack(fill='x', expand=True, padx=(5, 0))

        # Question input
        question_frame = ttk.Frame(self.root)
        question_frame.pack(pady=10, padx=10, fill='x')

        ttk.Label(question_frame, text="Question to ask the assistant:").pack(fill='x')
        self.question_entry = tk.Text(question_frame, height=4)
        self.question_entry.pack(fill='x', expand=True, padx=(5, 0))

        # Button frame for Submit and Reset
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        # Submit button
        self.submit_btn = ttk.Button(button_frame, text="Submit Question", command=self.generate_response)
        self.submit_btn.pack(side='left', padx=5)

        # Reset button
        self.reset_btn = ttk.Button(button_frame, text="Reset", command=self.reset_inputs)
        self.reset_btn.pack(side='left', padx=5)

        # Response area
        response_frame = ttk.Frame(self.root)
        response_frame.pack(pady=10, padx=10, fill='both', expand=True)

        self.response_text = tk.Text(response_frame, wrap='word', height=10)
        self.response_text.pack(fill='both', expand=True)

    def generate_response(self):
        # Clear previous response
        self.response_text.delete(1.0, tk.END)
        self.submit_btn.state(['disabled'])

        expertise = self.expertise_entry.get()
        question = self.question_entry.get("1.0", tk.END).strip()

        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text",
                                 "text": f"You are a helpful assistant who is an expert in the study of {expertise}"}, ]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}, ]
                },
            ],
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to("cpu")

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        thread = Thread(target=self.model.generate,
                        kwargs={"input_ids": inputs['input_ids'],
                                "streamer": streamer,
                                "max_new_tokens": 384,
                                "do_sample": True})
        thread.start()

        # Start monitoring the streamer in another thread
        Thread(target=self.process_stream, args=(streamer,)).start()
        self.check_queue()

    def process_stream(self, streamer):
        for new_text in streamer:
            self.response_queue.put(new_text)
        self.response_queue.put(None)  # Signal completion

    def check_queue(self):
        try:
            while True:
                text = self.response_queue.get_nowait()
                if text is None:  # Generation complete
                    self.submit_btn.state(['!disabled'])
                    break
                self.response_text.insert(tk.END, text)
                self.response_text.see(tk.END)
        except queue.Empty:
            self.root.after(100, self.check_queue)

    def reset_inputs(self):
        self.expertise_entry.delete(0, tk.END)
        self.question_entry.delete("1.0", tk.END)
        self.response_text.delete("1.0", tk.END)
        self.submit_btn.state(['!disabled'])

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = GemmaUI()
    app.run()
