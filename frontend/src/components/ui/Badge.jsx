import { cn } from '../../lib/utils'

const VARIANTS = {
  emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  red: 'bg-red-500/10 text-red-400 border-red-500/20',
  blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  slate: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
  brand: 'bg-brand-500/10 text-brand-400 border-brand-500/20',
}

export function Badge({ variant = 'slate', className, children, ...props }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold border tracking-wide uppercase',
        VARIANTS[variant] || VARIANTS.slate,
        className
      )}
      {...props}
    >
      {children}
    </span>
  )
}
