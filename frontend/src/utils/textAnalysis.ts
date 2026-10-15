import { TextStats } from '../types';

const SLANG_DICT = [
  'lol','lmao','lmfao','bruh','fr','ngl','tbh','idk','smh','omg','wtf',
  'irl','af','lowkey','highkey','slay','periodt','no cap','cap','bussin',
  'sus','salty','mood','vibe','yeet','fam','goat','based','cringe','ratio',
  'mid','understood the assignment','it hits different','glow up','finna',
  'bet','wya','gonna','wanna','kinda','sorta','nah','bro','bestie',
];

export function analyzeText(text: string): TextStats {
  const charCount = text.length;
  const wordCount = text.trim() === '' ? 0 : text.trim().split(/\s+/).length;
  const sentenceCount = (text.match(/[.!?]+/g) || []).length || (text.trim() !== '' ? 1 : 0);
  const hashtagCount = (text.match(/#\w+/g) || []).length;
  const mentionCount = (text.match(/@\w+/g) || []).length;
  const urlCount = (text.match(/https?:\/\/\S+|www\.\S+/gi) || []).length;
  const emojiCount = (text.match(/[\u{1F300}-\u{1FAD6}]|[\u{2600}-\u{27BF}]/gu) || []).length;
  const exclamationCount = (text.match(/!/g) || []).length;
  const questionCount = (text.match(/\?/g) || []).length;

  const words = text.trim().split(/\s+/);
  const capsWordCount = words.filter(w => w.length > 2 && w === w.toUpperCase() && /[A-Z]/.test(w)).length;
  const repeatedPunctCount = (text.match(/[!?]{2,}|\.{3,}/g) || []).length;

  const lowerText = text.toLowerCase();
  const possibleSlang = SLANG_DICT.filter(s => lowerText.includes(s));

  return {
    charCount,
    wordCount,
    sentenceCount,
    hashtagCount,
    mentionCount,
    urlCount,
    emojiCount,
    exclamationCount,
    questionCount,
    capsWordCount,
    repeatedPunctCount,
    possibleSlang: [...new Set(possibleSlang)],
  };
}

export function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ');
}

export function formatPercent(value: number, decimals = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatNumber(n: number): string {
  return new Intl.NumberFormat('en-IN').format(n);
}
