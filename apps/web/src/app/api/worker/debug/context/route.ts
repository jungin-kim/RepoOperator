import { NextResponse } from "next/server";

import { WorkerProxyError, workerProxyFetch } from "@/lib/worker-proxy";

export async function GET() {
  try {
    const response = await workerProxyFetch("/debug/context", { method: "GET" });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch (error) {
    if (error instanceof WorkerProxyError) return NextResponse.json({ detail: error.message }, { status: error.status });
    return NextResponse.json({ detail: "Unexpected error while loading context debug data." }, { status: 500 });
  }
}
