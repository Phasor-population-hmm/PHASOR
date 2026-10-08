import { notFound, redirect } from "next/navigation";
import { stageById } from "@/lib/stages";
import StageRunner from "@/components/StageRunner";

export default async function StagePage({ params }) {
  const { id } = await params;
  if (id === "hmm") redirect("/hmm");
  const stage = stageById(id);
  if (!stage) notFound();
  return <StageRunner stage={stage} />;
}
