import { protectedProcedure, router } from "../../index";
import { chatInput, reviewTextInput } from "./model";
import { chat, reviewText } from "./service";

export const aiRouter = router({
  chat: protectedProcedure
    .input(chatInput)
    .mutation(({ input }) => chat(input)),
  reviewText: protectedProcedure
    .input(reviewTextInput)
    .mutation(({ input }) => reviewText(input)),
});
