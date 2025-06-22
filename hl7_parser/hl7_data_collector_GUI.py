# record_hl7_with_exceptions.py
from __future__ import annotations
import asyncio, datetime, sys
from pathlib import Path
from hl7.mllp import start_hl7_server

class HL7Error(RuntimeError): ...

def _append(path: Path, msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    path.write_text(f"{path.read_text('utf-8') if path.exists() else ''}"
                    f"Frame timestamp: {ts}\nHL7 data:\n{msg}\n\n",
                    encoding="utf-8")


def record_hl7(start_evt,
               stop_evt,
               duration: int,
               out_dir: str | Path,
               port: int = 8008):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / "hl7_raw.txt"

    #  sync part
    if not start_evt.wait(timeout=30):
        raise HL7Error("[HL7] start_evt timeout (GUI)")

    #  asyncio part
    async def _main():
        async def _handler(reader, writer):
            try:
                while not writer.is_closing():
                    msg = await reader.readmessage()
                    _append(outfile, str(msg))
                    writer.writemessage(msg.create_ack()); await writer.drain()
            except asyncio.IncompleteReadError:
                pass
            finally:
                writer.close(); await writer.wait_closed()

        server = await start_hl7_server(_handler, port=port)
        print(f"[HL7] server on port {port}")

        async def wait_stop_evt():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, stop_evt.wait)

        try:
            await asyncio.wait(
                {asyncio.create_task(server.serve_forever()),
                 asyncio.create_task(wait_stop_evt())},
                timeout=duration,
                return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            server.close(); await server.wait_closed()
            print("[HL7] server closed")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_main())
    finally:
        loop.close()
