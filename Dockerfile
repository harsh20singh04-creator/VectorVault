FROM eclipse-temurin:21-jdk AS build
WORKDIR /app
COPY Main.java .
COPY src ./src
RUN mkdir -p out && find src -name '*.java' -print > sources.list && echo Main.java >> sources.list && javac --release 17 --add-modules jdk.httpserver -encoding UTF-8 -d out @sources.list

FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=build /app/out ./out
COPY index.html ./index.html
RUN mkdir -p /app/data && chown -R 10001:10001 /app
USER 10001:10001
ENV VECTORVAULT_HOST=0.0.0.0
ENV PORT=8080
EXPOSE 8080
VOLUME ["/app/data"]
CMD ["java", "--add-modules", "jdk.httpserver", "-Dvectorvault.root=/app", "-cp", "out", "Main"]
