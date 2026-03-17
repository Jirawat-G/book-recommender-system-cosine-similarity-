import { useState } from "react";

export default function App() {
  const [query, setQuery] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const search = async () => {
    setError("");
    setData(null);

    if (!query.trim()) {
      setError("กรุณากรอกคำค้น");
      return;
    }

    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/recommendations", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query,
          top_n: 5,
          top_k_classes: 2
        })
      });

      const json = await res.json();
      setData(json);
    } catch (e) {
      setError("เรียก API ไม่สำเร็จ");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 900, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>ระบบแนะนำหนังสือ</h1>

      <div style={{ display: "flex", gap: 8 }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="พิมพ์คำค้น เช่น python, sql, network"
          style={{ flex: 1, padding: 10 }}
        />
        <button onClick={search} style={{ padding: "10px 16px" }}>
          ค้นหา
        </button>
      </div>

      {error && <p style={{ color: "red" }}>{error}</p>}
      {loading && <p>กำลังค้นหา...</p>}

      {data && (
        <div style={{ marginTop: 24 }}>
          <h3>หมวดที่ระบบทำนาย</h3>
          <ul>
            {data.predicted_classes?.map((item) => (
              <li key={item.course_id}>
                {item.course_name} ({item.score.toFixed(2)})
              </li>
            ))}
          </ul>

          <h3>ผลลัพธ์</h3>
          {data.results?.map((book) => (
            <div
              key={book.book_id}
              style={{
                border: "1px solid #ddd",
                borderRadius: 8,
                padding: 16,
                marginBottom: 12
              }}
            >
              <h4>{book.title}</h4>
              <p><strong>ผู้แต่ง:</strong> {book.author || "-"}</p>
              <p><strong>หมวด:</strong> {book.course_name}</p>
              <p><strong>คะแนน:</strong> {book.score.toFixed(3)}</p>
              <p>{book.description}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}