import aiorun
import asyncio
import os
import datetime
import hl7
from hl7.mllp import start_hl7_server
import sys

OUTPUT_DIR = r"C:/Users/admin/Desktop/phd-workspace/input_data_10_06/HL7_raw_data_2/u130vswir"
os.makedirs(OUTPUT_DIR, exist_ok=True)
capture_duration = 5
#capture_duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
def generate_filename():
    """Generates a unique filename based on the timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    return os.path.join(OUTPUT_DIR, f"{timestamp}.txt")

OUTPUT_FILE = generate_filename()

async def append_to_file(fname, data):
    """Saves the HL7 message to a file with a timestamp for each frame."""
    frame_ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


    with open(fname, "a", encoding="utf-8") as f:
        f.write(f"Frame timestamp: {frame_ts_str}\n")
        f.write("HL7 Data:\n")
        f.write(str(data))
        f.write("\n\n")

async def process_hl7_messages(hl7_reader, hl7_writer):
    """Handles new connections and processes HL7 messages."""
    peername = hl7_writer.get_extra_info("peername")
    print(f"Connection established {peername}")

    try:
        while not hl7_writer.is_closing():
            hl7_message = await hl7_reader.readmessage()
            print(f"Received message\n {hl7_message}".replace('\r', '\n'))

            await append_to_file(OUTPUT_FILE, hl7_message)

            hl7_writer.writemessage(hl7_message.create_ack())
            await hl7_writer.drain()

    except asyncio.IncompleteReadError:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        if not hl7_writer.is_closing():
            hl7_writer.close()
            await hl7_writer.wait_closed()
        print(f"Connection closed {peername}")

async def main():
    """Starts the HL7 server and listens for connections."""
    server = await start_hl7_server(process_hl7_messages, port=8008)
    print("HL7 server started on port 8008.")

    server_task = asyncio.create_task(server.serve_forever())
    print("Server is listening...")

    await asyncio.sleep(capture_duration)

    print("Time is up – stopping the server...")
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        print("Server task was canceled.")

    print("HL7 server has been shut down. Program terminated.")

if __name__ == "__main__":
    asyncio.run(main())
