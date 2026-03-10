import winston from "winston";

const consoleTransport = new winston.transports.Console({
  stderrLevels: ["error", "warn", "info", "http", "verbose", "debug", "silly"],
  format: winston.format.combine(
    winston.format.colorize(),
    winston.format.printf(({ timestamp, level, message, ...meta }) => {
      let metaStr = "";
      if (Object.keys(meta).length) {
        try {
          metaStr = ` ${JSON.stringify(meta)}`;
        } catch {
          metaStr = " [unserializable meta]";
        }
      }
      return `${String(timestamp)} ${level}: ${String(message)}${metaStr}`;
    }),
  ),
});

export const logger = winston.createLogger({
  level: process.env.LOG_LEVEL ?? "info",
  format: winston.format.combine(winston.format.timestamp(), winston.format.json()),
  transports: [consoleTransport],
});
