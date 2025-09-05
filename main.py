import logging
import sys

# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("chatbot")

def main():
    logger.info("Hello from chatbot!")

if __name__ == "__main__":
    main()
