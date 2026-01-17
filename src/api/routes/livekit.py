import datetime
import os

from fastapi import APIRouter, HTTPException
from livekit import api
from pydantic import BaseModel

router = APIRouter()


class LivekitTokenRequest(BaseModel):
    identity: str
    room: str
    ttl_seconds: int | None = None
    dispatch: bool = False
    agent_name: str | None = None


@router.post("/livekit-token")
async def livekit_token(body: LivekitTokenRequest):
    api_key = os.getenv("LIVEKIT_API_KEY", "").strip()
    api_secret = os.getenv("LIVEKIT_API_SECRET", "").strip()
    livekit_url = os.getenv("LIVEKIT_URL", "").strip()
    if not api_key or not api_secret or not livekit_url:
        raise HTTPException(status_code=500, detail="LIVEKIT_URL/API_KEY/API_SECRET is not set")

    if not livekit_url.startswith(("http://", "https://", "ws://", "wss://")):
        livekit_url = f"wss://{livekit_url}"

    token = (
        api.AccessToken(api_key=api_key, api_secret=api_secret)
        .with_identity(body.identity)
        .with_name(body.identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=body.room,
                can_update_own_metadata=True,
            )
        )
    )
    if body.ttl_seconds:
        token = token.with_ttl(datetime.timedelta(seconds=int(body.ttl_seconds)))
    token = token.to_jwt()

    if body.dispatch:
        agent_name = (body.agent_name or os.getenv("LIVEKIT_AGENT_NAME", "")).strip()
        await _dispatch_agent(livekit_url, api_key, api_secret, body.room, agent_name)

    return {
        "token": token,
        "url": livekit_url,
        "livekit_url": livekit_url,
        "room": body.room,
        "identity": body.identity,
    }


async def _dispatch_agent(url: str, api_key: str, api_secret: str, room: str, agent_name: str) -> None:
    lk = api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret)
    try:
        req = api.agent_dispatch_service.CreateAgentDispatchRequest(room=room, agent_name=agent_name)
        await lk.agent_dispatch.create_dispatch(req)
    finally:
        await lk.aclose()
