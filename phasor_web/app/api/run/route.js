import { NextResponse } from "next/server";
import { stageById } from "@/lib/stages";
import { buildArgv, launch } from "@/lib/jobs";
import { pipelinePython } from "@/lib/pythonPath";

export async function POST(request) {
  const body = await request.json();
  const stage = stageById(body.stage);
  if (!stage) return NextResponse.json({ error: "unknown stage" }, { status: 400 });

  const values = body.values || {};
  const missing = stage.fields.filter((f) => f.required && !values[f.key]).map((f) => f.label);
  if (missing.length) return NextResponse.json({ error: "missing: " + missing.join(", ") }, { status: 400 });

  const argv = buildArgv(stage, values, pipelinePython());
  try {
    const job = launch(stage.title, argv, {
      output: values.out || values.cond || values.run || values.data || "",
      href: `/stage/${stage.id}`,
      stage: stage.id,
    });
    return NextResponse.json({ job: job.id });
  } catch (err) {
    if (err.busy) return NextResponse.json({ error: "Already one job or operation in process.", busy: true }, { status: 409 });
    throw err;
  }
}
