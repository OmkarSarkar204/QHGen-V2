import sys
import re
import datetime

class ConsoleRedirector:
    """
    Redirects stdout/stderr to a GUI text widget and sanitizes emojis for a professional look.
    """
    def __init__(self, text_widget, status_label=None):
        self.text_widget = text_widget
        self.status_label = status_label
        self.emoji_pattern = re.compile(r'[^\x00-\x7F]+') # Matches non-ASCII (emojis)

    def write(self, message):
        if not message: return
        
        # 1. Sanitize: Remove emojis
        clean_msg = self.emoji_pattern.sub('', message)
        
        # 2. Reformat specific tags for readability
        if "TOP CANDIDATE" in clean_msg:
            clean_msg = f"\n[DISCOVERY] {clean_msg.strip()}"
        elif "Quantum Truth" in clean_msg:
            clean_msg = f"[VALIDATION] {clean_msg.strip()}"
        elif "AI Prediction" in clean_msg:
            clean_msg = f"[PREDICTION] {clean_msg.strip()}"
            
        # 3. Timestamping for log lines (skip empty newlines)
        if clean_msg.strip():
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            final_msg = f"[{timestamp}] {clean_msg}"
        else:
            final_msg = clean_msg

        # 4. Update GUI (Thread safe update would be ideal, but direct works for simple apps)
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", final_msg)
        self.text_widget.see("end") # Auto-scroll
        self.text_widget.configure(state="disabled")

        # 5. Update Status Label if it's a major event
        if self.status_label and "[DISCOVERY]" in final_msg:
             self.status_label.configure(text="Status: Analyzing New Candidate...")

    def flush(self):
        pass