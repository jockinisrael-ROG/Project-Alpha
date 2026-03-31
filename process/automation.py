import datetime
import os
import subprocess
import webbrowser


def handle_automation(text: str) -> str | None:
    command = text.strip().lower()
    if not command:
        return None

    def has_any(*phrases: str) -> bool:
        return any(p in command for p in phrases)

    def wants_action() -> bool:
        return has_any("open", "launch", "start", "run", "go to")

    # Quick information tasks.
    if has_any("time", "what time", "current time"):
        now = datetime.datetime.now().strftime("%I:%M %p")
        return f"It is {now}."

    if has_any("date", "today's date", "current date"):
        today = datetime.datetime.now().strftime("%A, %d %B %Y")
        return f"Today is {today}."

    # Search shortcuts.
    if "search google for " in command:
        query = command.split("search google for ", 1)[1].strip()
        if query:
            webbrowser.open(f"https://www.google.com/search?q={query.replace(' ', '+')}")
            return f"Searching Google for {query}."

    if "search youtube for " in command:
        query = command.split("search youtube for ", 1)[1].strip()
        if query:
            webbrowser.open(
                f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            )
            return f"Searching YouTube for {query}."

    # Common websites.
    if (has_any("youtube") and wants_action()) or command == "open youtube":
        webbrowser.open("https://www.youtube.com")
        return "Opening YouTube now."

    if (has_any("google") and wants_action()) or command == "open google":
        webbrowser.open("https://www.google.com")
        return "Opening Google for you."

    if (has_any("gmail") and wants_action()) or command == "open gmail":
        webbrowser.open("https://mail.google.com")
        return "Opening Gmail."

    if (has_any("github") and wants_action()) or command == "open github":
        webbrowser.open("https://github.com")
        return "Opening GitHub."

    if (has_any("chatgpt", "openai") and wants_action()) or command == "open chatgpt":
        webbrowser.open("https://chat.openai.com")
        return "Opening ChatGPT."

    if (has_any("whatsapp") and wants_action()) or command == "open whatsapp":
        webbrowser.open("https://web.whatsapp.com")
        return "Opening WhatsApp Web."

    if (has_any("netflix") and wants_action()) or command == "open netflix":
        webbrowser.open("https://www.netflix.com")
        return "Opening Netflix."

    if (has_any("stack overflow", "stackoverflow") and wants_action()) or command == "open stackoverflow":
        webbrowser.open("https://stackoverflow.com")
        return "Opening Stack Overflow."

    if has_any("weather") and wants_action():
        webbrowser.open("https://www.google.com/search?q=weather")
        return "Opening weather forecast."

    # Windows apps and utilities.
    if (has_any("notepad") and wants_action()) or command == "open notepad":
        if os.name == "nt":
            subprocess.Popen(["notepad.exe"])
            return "Opening Notepad."
        return "Notepad is only available on Windows."

    if (has_any("calculator", "calc") and wants_action()) or command == "open calculator":
        if os.name == "nt":
            subprocess.Popen(["calc.exe"])
            return "Opening Calculator."
        return "Calculator shortcut is configured for Windows."

    if (has_any("command prompt", "cmd") and wants_action()) or command == "open cmd":
        if os.name == "nt":
            subprocess.Popen(["cmd.exe"])
            return "Opening Command Prompt."
        return "Command Prompt shortcut is configured for Windows."

    if (has_any("task manager") and wants_action()) or command == "open task manager":
        if os.name == "nt":
            subprocess.Popen(["taskmgr.exe"])
            return "Opening Task Manager."
        return "Task Manager shortcut is configured for Windows."

    if has_any("file explorer") and wants_action():
        if os.name == "nt":
            subprocess.Popen(["explorer.exe"])
            return "Opening File Explorer."
        return "File Explorer shortcut is configured for Windows."

    if has_any("settings") and wants_action():
        if os.name == "nt":
            subprocess.Popen(["start", "ms-settings:"], shell=True)
            return "Opening Settings."
        return "Settings shortcut is configured for Windows."

    if has_any("downloads") and wants_action():
        folder = os.path.join(os.path.expanduser("~"), "Downloads")
        if os.path.isdir(folder):
            if os.name == "nt":
                os.startfile(folder)
            else:
                webbrowser.open(f"file://{folder}")
            return "Opening Downloads folder."

    if has_any("documents") and wants_action():
        folder = os.path.join(os.path.expanduser("~"), "Documents")
        if os.path.isdir(folder):
            if os.name == "nt":
                os.startfile(folder)
            else:
                webbrowser.open(f"file://{folder}")
            return "Opening Documents folder."

    if has_any("desktop") and wants_action():
        folder = os.path.join(os.path.expanduser("~"), "Desktop")
        if os.path.isdir(folder):
            if os.name == "nt":
                os.startfile(folder)
            else:
                webbrowser.open(f"file://{folder}")
            return "Opening Desktop folder."

    if has_any("open chrome"):
        if os.name == "nt":
            try:
                subprocess.Popen(["chrome.exe"])
                return "Opening Chrome."
            except Exception:
                webbrowser.open("https://www.google.com")
                return "Chrome command was not found, opening Google in your default browser."

    if has_any("open edge"):
        if os.name == "nt":
            try:
                subprocess.Popen(["msedge.exe"])
                return "Opening Edge."
            except Exception:
                webbrowser.open("https://www.bing.com")
                return "Edge command was not found, opening Bing in your default browser."

    if has_any("open vscode", "open visual studio code"):
        if os.name == "nt":
            try:
                subprocess.Popen(["code"])
                return "Opening Visual Studio Code."
            except Exception:
                return "VS Code command 'code' was not found in PATH."

    return None
