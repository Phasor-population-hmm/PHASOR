"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export function useJobPoll(onDone) {
  const [lines, setLines] = useState([]);
  const [status, setStatus] = useState("idle");
  const [meta, setMeta] = useState(null);
  const timerRef = useRef(null);

  const stop = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
  }, []);

  const watch = useCallback(
    (jobId) => {
      stop();
      setLines([]);
      setStatus("running");
      timerRef.current = setInterval(async () => {
        const res = await fetch(`/api/job/${jobId}`);
        const j = await res.json();
        setLines(j.log || []);
        if (j.status !== "running") {
          setStatus(j.status);
          setMeta(j);
          stop();
          onDone && onDone(j);
        }
      }, 700);
    },
    [onDone, stop]
  );

  const reset = useCallback(() => {
    stop();
    setLines([]);
    setStatus("idle");
    setMeta(null);
  }, [stop]);

  useEffect(() => stop, [stop]);

  return { lines, status, meta, watch, reset, setLines };
}
