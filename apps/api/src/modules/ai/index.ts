import { protectedProcedure, router } from "../../index";
import { reviewTextInput } from "./model";
import { reviewText } from "./service";

export const aiRouter = router({
  reviewText: protectedProcedure
    .input(reviewTextInput)
    .mutation(({ input }) => reviewText(input)),
});
