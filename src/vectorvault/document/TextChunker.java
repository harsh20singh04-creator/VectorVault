package vectorvault.document;

import java.util.*;

public final class TextChunker {
    private TextChunker() {}

    public static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String trimmed = text == null ? "" : text.trim();
        if (trimmed.isEmpty()) return new ArrayList<>();
        String[] words = trimmed.split("\\s+");
        if (words.length <= chunkWords) return new ArrayList<>(List.of(text));

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        if (step <= 0) throw new IllegalArgumentException("overlapWords must be < chunkWords");
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            chunks.add(String.join(" ", Arrays.copyOfRange(words, i, end)));
            if (end == words.length) break;
        }
        return chunks;
    }
}
