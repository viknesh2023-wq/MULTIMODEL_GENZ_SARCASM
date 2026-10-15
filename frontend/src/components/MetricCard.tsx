import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

interface MetricCardProps {
  label: string;
  value: number;
  suffix?: string;
  isPercent?: boolean;
  color?: 'blue' | 'purple' | 'green' | 'amber';
  delay?: number;
  description?: string;
}

export function MetricCard({ label, value, suffix = '', isPercent = false, delay = 0, description }: MetricCardProps) {
  const [displayed, setDisplayed] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      const target = isPercent ? value * 100 : value;
      const duration = 1200;
      const step = 16;
      const increment = target / (duration / step);
      let current = 0;
      const interval = setInterval(() => {
        current = Math.min(current + increment, target);
        setDisplayed(current);
        if (current >= target) clearInterval(interval);
      }, step);
      return () => clearInterval(interval);
    }, delay * 1000);
    return () => clearTimeout(timer);
  }, [value, isPercent, delay]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
      className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 card-hover"
    >
      <div className="text-[11px] text-zinc-400 uppercase tracking-wider mb-2 font-medium">{label}</div>
      <div className="text-3xl font-bold font-mono text-white">
        {isPercent ? displayed.toFixed(4) : Math.round(displayed).toLocaleString('en-IN')}
        {isPercent ? '%' : suffix}
      </div>
      {description && <div className="text-xs text-zinc-500 mt-2">{description}</div>}
    </motion.div>
  );
}
