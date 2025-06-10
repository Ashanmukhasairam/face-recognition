import sys
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import os

class Watcher:
    def __init__(self, directory="."):
        self.observer = Observer()
        self.directory = directory
        self.process = None

    def run(self):
        """Method to start watching the directory"""
        event_handler = Handler(self)
        self.observer.schedule(event_handler, self.directory, recursive=True)
        self.observer.start()
        print(f"\n👀 Watching for file changes in {self.directory}")
        try:
            self.restart_program()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
            if self.process:
                self.process.terminate()
            print("\n🛑 Stopping the application...")
        self.observer.join()

    def restart_program(self):
        """Restarts the main program"""
        if self.process:
            self.process.terminate()
        print("\n🔄 Starting the application...")
        # Replace 'app.py' with your main application file
        self.process = subprocess.Popen([sys.executable, 'app.py'])

class Handler(FileSystemEventHandler):
    def __init__(self, watcher):
        self.watcher = watcher
        self.last_modified = time.time()

    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            # Add a small delay to prevent multiple reloads
            current_time = time.time()
            if (current_time - self.last_modified) > 1:
                print(f"\n📝 File changed: {event.src_path}")
                self.watcher.restart_program()
                self.last_modified = current_time

if __name__ == "__main__":
    directory = os.path.dirname(os.path.abspath(__file__))
    w = Watcher(directory)
    w.run()