package vectorvault.db;

import java.util.ArrayList;
import java.util.List;

public class DocItem {
    public int id; public String title, text; public List<Float> emb;
    public DocItem(int id, String title, String text, List<Float> emb) {
        this.id=id; this.title=title; this.text=text; this.emb=new ArrayList<>(emb);
    }
}
