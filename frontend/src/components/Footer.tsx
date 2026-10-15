export function Footer() {
  return (
    <footer className="border-t border-zinc-800 bg-black px-6 py-4 mt-12">
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-zinc-500">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-zinc-300">
            Multimodal Gen-Z Sarcasm Detection Research Project
          </span>
          <span>•</span>
          <span>Review 2 / Thesis Platform</span>
        </div>
        <div className="font-mono text-[11px] text-zinc-500">
          XLM-R Text Model • CLIP • InternVL3-2B • Multimodal Fusion
        </div>
      </div>
    </footer>
  );
}
