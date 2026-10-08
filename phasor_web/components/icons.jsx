const base = { fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round", strokeLinejoin: "round" };

export function IconWave(props) {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...base} {...props}>
      <path d="M2 12h3l2-7 3 14 3-14 2 7h3" />
      <path d="M19 12h3" />
    </svg>
  );
}

export function IconFilter(props) {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...base} {...props}>
      <path d="M4 5h16" />
      <path d="M7 12h10" />
      <path d="M10 19h4" />
    </svg>
  );
}

export function IconSine(props) {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...base} {...props}>
      <path d="M2 12c1.5-4 3.5-6 5.5-6s3 4 5 4 3.5-4 5.5-4 3.5 2 4 2" />
      <circle cx="12" cy="12" r="9" opacity=".35" />
    </svg>
  );
}

export function IconNetwork(props) {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...base} {...props}>
      <circle cx="5" cy="6" r="2" />
      <circle cx="5" cy="18" r="2" />
      <circle cx="12" cy="12" r="2" />
      <circle cx="19" cy="7" r="2" />
      <circle cx="19" cy="17" r="2" />
      <path d="M6.7 7.2L10.3 11M6.7 16.8L10.3 13M13.7 11L17.3 8M13.7 13L17.3 16" />
    </svg>
  );
}

export function IconChart(props) {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...base} {...props}>
      <path d="M4 20V10M12 20V4M20 20v-7" />
    </svg>
  );
}

export function IconArrowRight(props) {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" {...base} {...props}>
      <path d="M4 12h15M13 6l6 6-6 6" />
    </svg>
  );
}
