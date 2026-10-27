export const money = (v, cur = "US Dollar") =>
  `${Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${cur === "US Dollar" ? "USD" : cur}`;
export const pct = (v) => `${(Number(v) * 100).toFixed(0)}%`;
export const num = (v, d = 3) => (v == null ? "—" : Number(v).toFixed(d));
export const shortTime = (t) => String(t).slice(5, 16).replace("T", " ");
