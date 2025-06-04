import sys
import train

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train]")
        exit()

    cmd = sys.argv[1]
    if cmd == "train":
        train.train()
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()