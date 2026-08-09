import styles from './Button.module.css';

export default function Button({
  variant = 'default',
  type = 'button',
  className = '',
  children,
  ...rest
}) {
  const classes = [
    styles.btn,
    variant === 'primary' ? styles.primary : '',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <button type={type} className={classes} {...rest}>
      {children}
    </button>
  );
}
