import { createServer } from "node:http";
import { logger } from "./logger.js";
import { makeHandler } from "./server.js";

const port = Number(process.env.PORT ?? 8080);

const server = createServer(makeHandler());
server.listen(port, "0.0.0.0", () => {
  logger.info(`[nuke] worker-runtime listening on :${port}`, { port });
});
